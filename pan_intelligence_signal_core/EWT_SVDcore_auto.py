import numpy as np
import json
import csv
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from ewtpy import EWT1D

from scipy.signal import hilbert,windows,welch
from collections import OrderedDict
import scipy as sp
import os
from datetime import timezone
import pandas as pd
from scipy.signal import find_peaks

from scipy.io import loadmat

from sklearn.cluster import DBSCAN
from pan_intelligence_signal_core.signal_processor import init_signal_processor
from pan_intelligence_signal_core.sigma_bar_dao import SigmaBarDAO
from typing import List, Optional, Tuple, Dict, Any
import pymysql

INITIAL_THRESHOLDS = {
    "X": 5,
    "Y": 5,
    "Z": 5
}

def pad_to_same_length(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """统一填充函数，将两个一维数组补零到相同长度"""
    max_len = max(len(a), len(b))
    a_padded = np.pad(a, (0, max_len - len(a)), mode='constant')
    b_padded = np.pad(b, (0, max_len - len(b)), mode='constant')
    return a_padded, b_padded

class EWTProcessor:

    class AdaptiveEnvelopeDetector:
        """自适应包络检测器（保持内部类结构）"""
        def __init__(self, fs):
            self.fs = fs
            self.nfft = 4096
            self.freqs = None
            self.psd = None
            self.bands = []
            self.main_freqs = []

        def _compute_spectrum(self, signal):
            self.freqs, self.psd = welch(signal, self.fs, nperseg=self.nfft)
            return self.psd

        def _peak_detection(self):
            noise_floor = np.percentile(self.psd, 25)
            threshold = max(3*noise_floor, np.percentile(self.psd, 95))
            
            peaks, props = find_peaks(
                self.psd, 
                height=threshold,
                prominence=0.5*np.max(self.psd),
                distance=self.fs//100,
                width=True
            )
            return peaks, props

        def _frequency_clustering(self, peaks):
            df = np.mean(np.diff(self.freqs))
            X = self.freqs[peaks].reshape(-1, 1)
            db = DBSCAN(eps=2*df, min_samples=1).fit(X)
            
            clusters = []
            for label in np.unique(db.labels_):
                cluster_peaks = peaks[db.labels_ == label]
                main_peak = cluster_peaks[np.argmax(self.psd[cluster_peaks])]
                clusters.append(main_peak)
            return np.array(clusters)

        def process(self, signal):
            """实现标准化的process接口"""
            self._compute_spectrum(signal)
            peaks, props = self._peak_detection()
            
            if len(peaks) == 0:
                return {"bands": [], "main_freqs": []}
            
            main_peaks = self._frequency_clustering(peaks)
            df = np.mean(np.diff(self.freqs))
            
            self.bands = []
            self.main_freqs = self.freqs[main_peaks]
            for p in main_peaks:
                idx = np.where(peaks == p)[0][0]
                width = props['widths'][idx] * df
                self.bands.append([
                    max(0, self.freqs[p]-3*width),
                    min(self.fs//2, self.freqs[p]+3*width)
                ])
            return {
                "bands": self.bands,
                "main_freqs": self.main_freqs,
                "psd": self.psd,          # 新增
                "freqs": self.freqs       # 新增
            }
        
    class EWTTransformer:
        """经验小波变换器"""
        def __init__(self, num_modes=None):
            self.num_modes = num_modes
            self.modes = None

        def process(self, 
               signal: np.ndarray,
               bands: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        # 显式计算模式数
            num_modes = len(bands) + 1 if bands is not None else 3  # 默认3个模式
            
            # 防御性检查
            if num_modes < 1:
                raise ValueError(f"无效模式数: {num_modes} (bands长度: {len(bands)})")
            
            # 执行EWT分解
            ewt_modes, _, _ = EWT1D(signal, N=num_modes)
            
            
            # 构建带频带注释的结果
            return {
                f"mode_{i+1}": {
                    "signal": ewt_modes[:, i],
                    "band": bands[i] if (bands and i < len(bands)) else None,
                    "main_freq": (
                        np.mean(bands[i]) 
                        if (bands and i < len(bands)) 
                        else None
                    )
                } for i in range(ewt_modes.shape[1])
            }

    class HilbertFeatureExtractor:
        """希尔伯特特征提取器"""
        def __init__(self, fs):
            self.fs = fs

        @staticmethod
        def normalize_signal(signal: np.ndarray) -> np.ndarray:
            """信号归一化（静态方法）"""
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        def _process_single_mode(self, signal: np.ndarray) -> dict:
            """处理单个模态"""
            analytic_signal = hilbert(signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            
            return {
                'instantaneous_amp': np.abs(analytic_signal),
                'instantaneous_freq': (
                    np.append(np.diff(instantaneous_phase), 0) / (2 * np.pi) * self.fs
                )
            }
        
        def process(self, ewt_result: dict) -> np.ndarray:
            """返回形状为 (n_samples, n_imf) 的瞬时幅值矩阵"""
            imf_amplitudes = []
            for imf_name, imf_data in ewt_result.items():
                analytic_signal = hilbert(imf_data['signal'])
                amplitude = np.abs(analytic_signal)
                imf_amplitudes.append(amplitude)
            
                # amp_matrix = np.array(imf_amplitudes).T  # (n_samples, n_imf)

                # # 每列 z-score 标准化
                # amp_matrix -= np.mean(amp_matrix, axis=0)
                # amp_matrix /= (np.std(amp_matrix, axis=0) + 1e-8)  # 避免除以 0

            # 转置为 (时间点 × IMF分量)
            return np.array(imf_amplitudes).T

    class SVDAnalyzer:
        """增强版SVD分析器（支持基准计算）"""
        def __init__(self, sigma_bar: np.ndarray = None):
            self.sigma_bar = sigma_bar  # 正常状态基准
            
        @staticmethod
        def extract_amp_matrix(features: dict) -> np.ndarray:
            """从特征字典中提取幅值矩阵"""
            return np.column_stack([
                features[mode]['instantaneous_amp'] 
                for mode in sorted(features.keys(), key=lambda x: int(x.split('_')[1]))
            ])
        
        def _calculate_di(self, sigma: np.ndarray) -> float:
            """计算特征距离"""
            if self.sigma_bar is None:
                raise ValueError("必须先设置sigma_bar基准")
            sigma, sigma_bar = pad_to_same_length(sigma, self.sigma_bar)
            numerator = np.linalg.norm(sigma - sigma_bar,ord=2)
            denominator = np.linalg.norm(sigma_bar, ord=2)
            return numerator / denominator if denominator != 0 else 0.0
        
        def process(self, amp_matrix: np.ndarray, sigma_ref: np.ndarray) -> dict:
            """执行SVD分析"""
            
            U, s, Vt = np.linalg.svd(amp_matrix, full_matrices=False)

            # print("\n=== SVD分析 ===")
            print(f"当前奇异值: {s[:5].round(3)}... (共{len(s)}个)")
            # di = d_i = self._calculate_deviation(s, sigma_ref)
            if sigma_ref is not None:
                print(f"基准奇异值: {sigma_ref[:5].round(3)}... (共{len(sigma_ref)}个)")
                self.sigma_bar = sigma_ref
                di= self._calculate_di(s)
                print(f"DI值计算: {di:.4f}")
            else:
                di = 0.0
                return {
                    'singular_values': s,
                    'd_i': 0.0  # 首次计算时无基准，偏差为0
                }
            return {
                'singular_values': s,
                'd_i': di
            }
        
        def _calculate_deviation(self, s: np.ndarray, sigma_ref: np.ndarray) -> float:
            """计算当前奇异值与基准之间的偏差"""
            s_padded, ref_padded = pad_to_same_length(s, sigma_ref)
            return np.linalg.norm(s_padded - ref_padded)
    
        @classmethod
        def calculate_sigma_bar(cls, normal_features_list: list) -> np.ndarray:
            """计算正常状态基准（类方法）"""
            all_s = []
            for features in normal_features_list:
                amp_matrix = cls.extract_amp_matrix(features)
                amp_matrix -= np.mean(amp_matrix, axis=0)
                amp_matrix /= (np.std(amp_matrix, axis=0) + 1e-8)

                _, s, _ = np.linalg.svd(amp_matrix, full_matrices=False)
                all_s.append(s)
            return np.mean(all_s, axis=0)

    class SigmaBarUpdater:
        """独立Sigma基准维护模块"""
        def __init__(self, decay_factor: float = 0.5, axis_name: str = 'X', store_dir: str = './sigma_store'):
            """
            Args:
                decay_factor: 旧基准权重 (0.5=新旧平均，1.0=完全信任新数据)
            """
            self.axis_name = axis_name
            self.store_dir = store_dir
            self.sigma_bar: Optional[np.ndarray] = None
            self.decay_factor = decay_factor
            self.history: List[np.ndarray] = []
            self._load_sigma_bar()

        def update(self, new_data_features: List[np.ndarray]) -> None:
            """使用新数据特征增量更新 sigma_bar"""
            current_sigma = self._batch_compute(new_data_features)
            print("\n=== 基准更新 ===")
            if self.sigma_bar is None:
                print(f"初始化基准 | 长度: {len(current_sigma)}")
                self.sigma_bar = current_sigma
            else:
                padded_old, padded_new = pad_to_same_length(self.sigma_bar, current_sigma)
                self.sigma_bar = (
                    self.decay_factor * padded_old + (1 - self.decay_factor) * padded_new
                )
                
            self.history.append(self.sigma_bar.copy())
            self._save_sigma_bar()

        def _load_sigma_bar(self):
            path = os.path.join(self.store_dir, f'sigma_bar_{self.axis_name}.npy')
            if os.path.exists(path):
                self.sigma_bar = np.load(path)

        def _save_sigma_bar(self):
            os.makedirs(self.store_dir, exist_ok=True)
            path = os.path.join(self.store_dir, f'sigma_bar_{self.axis_name}.npy')
            np.save(path, self.sigma_bar)

        def reset(self) -> None:
            """重置 sigma_bar"""
            self.sigma_bar = None
            self.history.clear()
            path = os.path.join(self.store_dir, f'sigma_bar_{self.axis_name}.npy')
            if os.path.exists(path):
                os.remove(path)

        @staticmethod
        def _batch_compute(features_list: List[np.ndarray]) -> np.ndarray:
            """批量计算当前数据块的特征值均值（SVD奇异值）"""
            max_dim = max(min(f.shape) for f in features_list)
            all_s = []

            for features in features_list:
                _, s, _ = np.linalg.svd(features, full_matrices=False)
                s_padded = np.pad(s, (0, max_dim - len(s)), mode='constant')
                all_s.append(s_padded)

            return np.mean(all_s, axis=0)

        @property
        def current(self) -> np.ndarray:
            """获取当前基准"""
            if self.sigma_bar is None:
                raise ValueError("Sigma bar not initialized!")
            return self.sigma_bar.copy()
    
    class NumpyEncoder(json.JSONEncoder):
        """增强型数据类型处理"""
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return [round(float(x), 2) for x in obj.tolist()]
            if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return round(float(obj), 2)
            if isinstance(obj, (np.int_, np.intc, np.int32, np.int64)):
                return int(obj)
            return super().default(obj)

    class AnomalyDetector:
        """异常检测器，用于判断是否更新 sigma_bar"""
        def __init__(self):
            self.di_history = []  # 存储所有历史 d_i 值
            # self.threshold = None  # 动态阈值
           
        
        def add_di(self, di: float) -> None:
            """添加新的 d_i 值到历史记录"""
            self.di_history.append(di)
            # 更新阈值
            # self.threshold = np.mean(self.di_history)
            # print(f"[DEBUG] 当前 di_history: {self.di_history}")

        def is_anomaly(self, di: float,axis:str) -> bool:
            """判断当前 d_i 是否为异常值"""
            # 如果没有足够的历史数据，默认为正常
            if len(self.di_history) < 3:
                return False
            threshold = np.mean(self.di_history)
            initial_threshold = INITIAL_THRESHOLDS.get(axis, 30)
            # 如果 d_i 大于阈值，则认为是异常
            return di > threshold and di > initial_threshold
        
        def should_update_sigma_bar(self, di: float, axis: str) -> bool:
            """判断是否应该更新 sigma_bar"""
            # 添加当前 d_i 到历史记录
            is_abnomal = self.is_anomaly(di, axis)
            self.add_di(di)
            
            # 判断是否为异常
            return not is_abnomal
        
        def get_threshold(self) -> Optional[float]:
            """获取当前阈值"""
            return np.mean(self.di_history) if self.di_history else None
        
        def get_history(self) -> List[float]:
            """获取所有历史 d_i 值"""
            return self.di_history.copy()


    def __init__(self, csv_path=None, csv_string=None, decay_factor=0.5,db_config=None,serial_no="unknown"):
        SignalProcessorClass = init_signal_processor()
        self.sp = SignalProcessorClass()
        if csv_string:
            self.raw_data = self._parse_data(csv_string)
        elif csv_path:
            self.raw_data = self._parse_csv(csv_path)
        else:
            raise ValueError("csv_path 或 csv_string 必须指定")

        self.time_freq_data = self._process_time_freq()
        self.selected_axis = None
        self.results = {}
        self.sigma_updater = self.SigmaBarUpdater(decay_factor=decay_factor)
        self.sigma_updater_map = {
            axis: self.SigmaBarUpdater(decay_factor=decay_factor,axis_name=axis)
            for axis in ['X', 'Y', 'Z']
        }
        self.svd_analyzer = self.SVDAnalyzer()
        self.csv_path = csv_path

        self.sigma_bar_dao = SigmaBarDAO(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"]
        )

        self.anomaly_detectors = {
            axis: self.AnomalyDetector() 
            for axis in ['X', 'Y', 'Z']
        }
        self.serial_no = serial_no
        self.d_i_map = {}

    def _parse_csv(self, path):
        return self.sp.parse_data_from_csv_file(csv_file_path=path)
    
    def _parse_data(self,csv_str):
        return self.sp.parse_data(csv_data_str=csv_str)

    def _process_time_freq(self):
        return {
            axis: self.sp.time_freq_data_process(
                time_domain_curve_raw=self.raw_data[f"time_domain_curve_raw_{axis}"],
                scale_val=self.raw_data["scale_data"][axis],
                unit_after_scaling=self.raw_data["unit_after_scaling"],
                samp_freq=self.raw_data["sampling_freq"],
                code_window_type='hamming'
            ) for axis in ['X', 'Y', 'Z']
        }

    def batch_analysis(self, sensor_indices: list, normal_range: tuple = None):
        """支持增量学习的批量分析"""
        # 阶段1：基准更新（当传入normal_range时）
        if normal_range is not None:
            normal_features = []
            for idx in range(normal_range[0], normal_range[1]+1):
                self.analyze_axis(f'X_{idx}')
                normal_features.append(self.amp_matrix)
            
            # 增量更新基准值
            self.sigma_updater.update(normal_features)

        # 阶段2：异常检测
        final_results = {}
        for idx in sensor_indices:
            self.analyze_axis(f'X_{idx}')
            # 获取当前基准值并分析
            svd_result = self.svd_analyzer.process(
                features=self.hilbert_features,
                sigma_ref=self.sigma_updater.current  # 传入当前基准
            )
            final_results[f'sensor_{idx}'] = svd_result
        
        return final_results
    
    def analyze_axis(self, axis='X'):
        """执行完整分析流程"""
        self.selected_axis = axis
        signal = self.time_freq_data[axis]['time_domain']['normalized_val']
        
        # 包络检测
        envelope_detector = self.AdaptiveEnvelopeDetector(self.raw_data["sampling_freq"])
        envelope_result = envelope_detector.process(signal)
        
        # EWT分解
        ewt_transformer = self.EWTTransformer()
        ewt_result = ewt_transformer.process(signal, bands=envelope_result["bands"])
        
        # 希尔伯特特征提取
        hilbert_extractor = self.HilbertFeatureExtractor(fs=self.raw_data["sampling_freq"])
        amp_matrix = hilbert_extractor.process(ewt_result)
        self.amp_matrix = amp_matrix

        # 对应轴的 sigma_bar 更新器
        updater = self.sigma_updater_map[axis]
        current_sigma_ref = updater.current if updater.sigma_bar is not None else None

        if updater.sigma_bar is None:
            updater.update([amp_matrix])  # 初始化
            current_sigma_ref = updater.current  # 立即获取最新基准
        else:
            current_sigma_ref = updater.current
        # SVD分析
        svd_result = self.svd_analyzer.process(
            amp_matrix=amp_matrix,
            sigma_ref=updater.current
        )        
        current_di = svd_result['d_i']

        # 使用异常检测器判断是否应该更新 sigma_bar
        anomaly_detector = self.anomaly_detectors[axis]
        should_update = anomaly_detector.should_update_sigma_bar(current_di,axis)

        # 只有在应该更新时才进行更新
        if should_update:
            if updater.sigma_bar is None:
                # 初始化基准
                updater.update([amp_matrix])
            else:
                # 增量更新
                updater.update([amp_matrix])

        # 整合结果
        self.results[axis] = {
            "envelope": envelope_result,
            "ewt": ewt_result,
            "amp_matrix": self.amp_matrix,
            "svd": svd_result,
            "anomaly_detection": {
                "is_anomaly": not should_update,
                "threshold": anomaly_detector.get_threshold(),
                "di_history": anomaly_detector.get_history()
            }
        }
        return self

    def analyze_all_axes(self) -> None:
        """
        对 X/Y/Z 三个方向的信号执行完整分析流程，
        并在所有分析完成后写入 sigma_bar 和当前 d_i 到数据库
        """

        # 获取当前阈值
        thresholds = self.sigma_bar_dao.get_thresholds()
        if thresholds:
            # thresholds = [INITIAL_THRESHOLDS['X'], INITIAL_THRESHOLDS['Y'], INITIAL_THRESHOLDS['Z']]

            # 设置异常检测器的阈值
            for i, axis in enumerate(['X', 'Y', 'Z']):
                if thresholds[i] is not None:
                    self.anomaly_detectors[axis].threshold = thresholds[i]

                # 读取历史 d_i 数据并赋值给 anomaly_detector
                di_history = self.sigma_bar_dao.get_di_history(axis)  # 从数据库读取历史值
                if di_history:
                    self.anomaly_detectors[axis].di_history = di_history  # 更新历史值
        
        for axis in ['X', 'Y', 'Z']:
            self.analyze_axis(axis)
        
       
        # 只判断是否异常，不影响内部历史
        is_anomaly_x = self.anomaly_detectors['X'].is_anomaly(self.results['X']['svd']['d_i'], axis='X')
        is_anomaly_y = self.anomaly_detectors['Y'].is_anomaly(self.results['Y']['svd']['d_i'], axis='Y')
        is_anomaly_z = self.anomaly_detectors['Z'].is_anomaly(self.results['Z']['svd']['d_i'], axis='Z')

        
        # 插入 d_i 记录
        self.sigma_bar_dao.insert_di_record(
            serial_no=self.serial_no,
            x=self.results['X']['svd']['d_i'],
            y=self.results['Y']['svd']['d_i'],
            z=self.results['Z']['svd']['d_i'],
            is_anomaly_x=is_anomaly_x,
            is_anomaly_y=is_anomaly_y,
            is_anomaly_z=is_anomaly_z
        )
        
        # 如果没有异常，则更新 sigma_bar
        if not (is_anomaly_x or is_anomaly_y or is_anomaly_z):
            self.sigma_bar_dao.update_sigma_bar(
                x=self.sigma_updater_map['X'].sigma_bar,
                y=self.sigma_updater_map['Y'].sigma_bar,
                z=self.sigma_updater_map['Z'].sigma_bar
            )
        
        # 更新阈值
        self.sigma_bar_dao.update_thresholds(
            x_threshold=self.anomaly_detectors['X'].get_threshold(),
            y_threshold=self.anomaly_detectors['Y'].get_threshold(),
            z_threshold=self.anomaly_detectors['Z'].get_threshold()
        )

    def get_sigma_bar(self) -> List[float]:
        """获取当前 Sigma 基准值（适用于前端展示）"""
        try:
            return self.sigma_updater.current.tolist()
        except ValueError:
            return []

    def _serialize_signal(self, signal):
        """信号数据序列化优化"""
        return {
            "length": len(signal),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "mean": float(np.mean(signal)),
            "data": np.round(signal, 2).tolist()  # 保留4位小数
        }

    def get_anomaly_detection_results(self, axis='X') -> dict:
        """获取指定轴的异常检测结果"""
        if axis not in self.results or 'anomaly_detection' not in self.results[axis]:
            return {
                "is_anomaly": False,
                "threshold": None,
                "di_history": []
            }
        return self.results[axis]['anomaly_detection']

    def visualize(self):
        """定制化可视化方法"""
        if not self.results or self.selected_axis not in self.results:
            raise ValueError("请先执行 analyze_axis() 并选择有效轴")

        current_data = self.results[self.selected_axis]
        
        # 创建2行1列的布局
        plt.figure(figsize=(16, 12))
        
        # ----------------- 图1：频带划分图 -----------------
        ax1 = plt.subplot(2, 1, 1)
        envelope = current_data['envelope']
        
        # 绘制PSD曲线
        ax1.semilogy(envelope['freqs'], envelope['psd'], 
                    color='#2c3e50', linewidth=1.5, label='PSD')
        
        # 频带区域填充
        for i, band in enumerate(envelope['bands']):
            ax1.axvspan(band[0], band[1], alpha=0.2, 
                    color=['#e74c3c', '#3498db'][i%2],
                    label=f'Band {i+1}' if i < 2 else None)
        
        # 标记主频率
        ax1.scatter(envelope['main_freqs'], envelope['psd'][np.searchsorted(envelope['freqs'], envelope['main_freqs'])],
                color='#e67e22', zorder=5, s=80, edgecolor='white', label='Dominant Freq')
        
        ax1.set_title(f"Frequency Band Partitioning - {self.selected_axis} Axis", fontsize=14)
        ax1.set_xlabel("Frequency (Hz)", fontsize=12)
        ax1.set_ylabel("Power/Frequency (dB/Hz)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_xlim(0, self.raw_data["sampling_freq"]//2)

        # ----------------- 图2：EWT模态分析 -----------------
        ax2 = plt.subplot(2, 1, 2)
        ewt_modes = list(current_data['ewt'].items())[:2]  # 取前两个模态
        
        # 创建左右子图
        plt.subplots_adjust(hspace=0.5)
        axs = [ax2.inset_axes([0.05, 0.5, 0.4, 0.4]),   # 模态1时域
            ax2.inset_axes([0.55, 0.5, 0.4, 0.4]),  # 模态1频域
            ax2.inset_axes([0.05, 0.05, 0.4, 0.4]),  # 模态2时域
            ax2.inset_axes([0.55, 0.05, 0.4, 0.4])]  # 模态2频域

        colors = ['#2980b9', '#c0392b']
        
        for i, (mode_name, mode_data) in enumerate(ewt_modes):
            signal = mode_data['signal']
            fs = self.raw_data["sampling_freq"]
            
            # 时域图
            axs[i*2].plot(np.arange(len(signal))/fs, signal, 
                        color=colors[i], linewidth=1.2)
            axs[i*2].set_title(f"{mode_name} - Time Domain", fontsize=10)
            axs[i*2].set_xlabel("Time (s)", fontsize=8)
            axs[i*2].set_ylabel("Amplitude", fontsize=8)
            
            # 频域图
            freqs = np.fft.rfftfreq(len(signal), 1/fs)
            fft_vals = np.abs(np.fft.rfft(signal))
            axs[i*2+1].plot(freqs, fft_vals, color=colors[i], linewidth=1.2)
            axs[i*2+1].set_title(f"{mode_name} - Frequency Domain", fontsize=10)
            axs[i*2+1].set_xlabel("Frequency (Hz)", fontsize=8)
            axs[i*2+1].set_ylabel("Magnitude", fontsize=8)
            axs[i*2+1].set_xlim(0, fs//2)

        plt.suptitle("Empirical Wavelet Transform Modal Analysis", y=0.95, fontsize=14)
        plt.tight_layout()
        plt.show()

    def save_results(self, output_dir="results"):
        """修复空值问题的保存方法"""
        # 确保三轴数据完整
        if not all(axis in self.results for axis in ['X', 'Y', 'Z']):
            raise ValueError("请先执行 analyze_all_axes()")

        # 构建与图示完全一致的结构
        output = {
            "data": {
                axis: {
                    f"mode_{i+1}": {
                        "signal": np.round(mode['signal'], 4).tolist(),
                        "band": mode['band'] if mode['band'] else None,
                        # 空值处理关键修复点 ▼
                        "main_freq": float(mode['main_freq'] if mode['main_freq'] is not None else 0.0)
                    } for i, mode in enumerate(axis_data['ewt'].values())
                } for axis, axis_data in self.results.items()
            },
            "svd": {
                axis: {
                    "singular_values": axis_data['svd']['singular_values'].tolist(),
                    "d_i": float(axis_data['svd']['d_i'])
                } for axis, axis_data in self.results.items()
            }
        }

        # 保存文件
        utc_time = datetime.fromtimestamp(
                self.raw_data["timestamp"], 
                tz=timezone.utc
            )
        date_str = utc_time.strftime("%Y%m%d")
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        output_file = f"{date_str}_{base_name}_EWT_SVDprocess.json"
        output_path = os.path.join(output_dir, output_file)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4, cls=self.NumpyEncoder)
        
        print(f"结果已保存至：{output_path}")

    def _format_axis_data(self, axis: str) -> dict:
        """按格式封装单轴数据"""
        data = self.results[axis]
        return {
            f"mode_{i+1}": {
                "signal": mode_data['signal'],
                "band": mode_data['band'] or None,
                "main_freq": mode_data['main_freq']
            } for i, (mode_name, mode_data) in enumerate(data['ewt'].items())
        }
