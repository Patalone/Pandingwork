# -*- coding: utf-8 -*-
# ---------------------------------------------------------------
# author: Ash
# signature:
# ---------------------------------------------------------------
#
#            C\
#           y  c
#          jK  $
#          @   jc
#         j@    &
#         $&    Mw@g
#         @&   ,@@"%g
#         *M@gWMj&  My
#            @Mwg@   3
#           / _@M*~.,_'
#          /.<"      `
#          `
# ---------------------------------------------------------------
import os
import numpy as np
import time
import datetime
import yaml
from flask import Flask, jsonify, render_template, request, send_file
from flask_restx import Api, Resource
from flasgger import Swagger
from flasgger.utils import swag_from
from waitress import serve
from flask import current_app
import logging
from logging.config import dictConfig
from pan_intelligence_signal_core.signal_processor import init_signal_processor
from joblib import dump, load
import yaml
import platform
#================================================================================
import traceback
import logging
from werkzeug.utils import secure_filename
from pan_intelligence_signal_core.EWT_SVDcore_auto import EWTProcessor  # 主算法类
import json
from flask import Response
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，适合服务器环境
import matplotlib.pyplot as plt
import time
import hashlib
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
from logging.handlers import RotatingFileHandler
from threading import Lock
import pymysql
#===================================================================================
ABS_FILE = os.path.abspath(__file__)
ABS_PATH = os.path.dirname(ABS_FILE)
ABS_CONFIG_PATH = os.path.join(ABS_PATH, "config.yml")
ABS_SWAGGER_CONFIG_PATH = os.path.join(ABS_PATH, "api_config.yml")
ABS_LOCK_FILE_PATH = os.path.join(ABS_PATH, "signal_app.lock")

N_DECIMAL_PLACE = 8  # 返回浮点数保留的小数位数
#===================================================================================
with open('configs/api_settings.yml', 'r', encoding='utf-8') as f:  # 文档1的配置
    ewt_config = yaml.safe_load(f)
#===================================================================================
system = platform.system()
if system == "Windows":
    import msvcrt
    lock_file = open(ABS_LOCK_FILE_PATH, "w")
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except IOError:
        print("Another instance of the app_center is already running. Exiting.")
        exit()

def load_yaml_file(yaml_file_path: str):
    with open(yaml_file_path, "r", encoding="utf_8") as config_file:
        yaml_content = config_file.read()
    return yaml_content

def keep_specified_decimal(input_data: any, n_decimal_place: np.int64 = N_DECIMAL_PLACE) -> np.array:
    if type(input_data) is not np.ndarray:
        if input_data is None:
            return None
    return np.round(a=input_data, decimals=n_decimal_place)

def response_float_num_trans(data: np.float64) -> np.float64:
    if data is not None:
        return np.float64(keep_specified_decimal(input_data=data))
    else:
        return None

app_yaml_content = load_yaml_file(yaml_file_path=ABS_CONFIG_PATH)

# Replace placeholders with actual values
actual_size = 20 * 1024 * 1024  # 日志文件最大20M
app_yaml_content = app_yaml_content.replace("${placeholder}", str(actual_size))
config_data = yaml.safe_load(app_yaml_content)

log_config = config_data["signal_app_log_config"]
dictConfig(log_config)
#===================================================================================
# 合并配置
config_data.update({
    'UPLOAD_FOLDER': ewt_config['upload_dir'],
    'MAX_CONTENT_LENGTH': ewt_config['max_file_size'] * 1024 * 1024
})
#===================================================================================
SERVE_PORT = config_data["serve_port"]
CODE_SUCCESS = config_data["response_code"]["success"]
CODE_PREPARING = config_data["response_code"]["preparing"]
CODE_ERROR = config_data["response_code"]["error"]
PRINT_ENABLE = config_data["print_enable"]
LOG_ENABLE = config_data["log_enable"]

if system == "Windows":
    auto_start = config_data["auto_start"]
    shortcut_name = "pan_intelligence_signal_service"
    start_up_path = os.path.join(os.environ["APPDATA"], "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
    if not os.path.exists(start_up_path):
        # 适配windows server 2016
        start_up_path = os.path.join("C:\ProgramData", "Microsoft", "Windows", "Start Menu", "Programs", "StartUp")

    shortcut_path = os.path.join(start_up_path, f"{shortcut_name}.lnk")
    if auto_start:
        start_batch_path = os.path.join(ABS_PATH, "start_pan_intelligence_signal_app.bat")
        try:
            # Check if the shortcut file already exists
            if not os.path.exists(shortcut_path):
                import win32com.client
                # Create a shortcut to the batch file in the Startup folder
                shell = win32com.client.Dispatch("WScript.Shell")
                shortcut = shell.CreateShortcut(shortcut_path)
                shortcut.TargetPath = start_batch_path
                shortcut.Save()
                print(f"Shortcut '{shortcut_name}' created successfully in Startup folder.")
            else:
                print(f"Shortcut '{shortcut_name}' already exists in Startup folder.")
        except Exception as e:
            print(f"Error creating shortcut: {e}")
            raise e
    else:
        try:
            # Check if the shortcut file already exists
            if os.path.exists(shortcut_path):
                os.remove(shortcut_path)
                print(f"Shortcut '{shortcut_name}' deleted successfully in Startup folder.")
            else:
                print(f"Shortcut '{shortcut_name}' not exist in Startup folder.")
        except Exception as e:
            print(f"Error deleting shortcut: {e}")
            raise e

MESSAGE_ERR_NOT_VALID = "服务未取得授权或已失效，请联系供应商取得授权或延长服务时效"
Signal_Processor = init_signal_processor()
signal_processor = Signal_Processor()

# N_DECIMAL_PLACES = config_data["control_params"]["n_decimal_places"]

# app = Flask(__name__)
app = Flask("signal_app")

# Swagger(app)
app.config['UPLOAD_FOLDER'] = config_data.get('UPLOAD_FOLDER', './uploads')
swagger_yaml_content = load_yaml_file(yaml_file_path=ABS_SWAGGER_CONFIG_PATH)

swagger_config_data = yaml.safe_load(swagger_yaml_content)
app.config["SWAGGER"] = {
    "title": swagger_config_data["swagger_title"],
    "openapi": "3.0.2",
}
#===================================================================================
# ======================== 日志配置 ===========================
dictConfig(config_data["signal_app_log_config"])
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# ======================== 全局变量 ===========================
analyze_cache = {}  # 合并后的全局缓存
global_processor_map = {}
global_processor_lock = Lock()
Signal_Processor = init_signal_processor()
signal_processor = Signal_Processor()
#===================================================================================
swagger_config = Swagger.DEFAULT_CONFIG
swagger_config["title"] = swagger_config_data["swagger_title"]
swagger_config["description"] = swagger_config_data["swagger_desc"]
swagger_config["host"] = swagger_config_data["swagger_host"]
swagger_config["swagger_ui"] = swagger_config_data["swagger_on"]  # 开启或关闭swagger ui

# api = Api(app, version="1.0", title="Signal API", description="Signal API documentation")
swagger = Swagger(app, template_file="swagger_template.yml", config=swagger_config)

def run_app(app, port):
    print("start")
    serve(app=app, host="0.0.0.0", port=port)
    app.logger.name = "signal_app"
    socketHandler = logging.handlers.SocketHandler("localhost", logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    app.logger.addHandler(socketHandler)

def pretreat_request_data(request, request_name):
    request_data = request.get_json()
    if PRINT_ENABLE:
        print(f"request data in /{request_name}")
        print(request_data)
    if LOG_ENABLE:
        current_app.logger.info(f"request data in /{request_name}")
        current_app.logger.info(request_data)
    return request_data

def pretreat_response_data(response_data: dict, request_name: str) -> None:
    if PRINT_ENABLE:
        print(f"response data in /{request_name}")
        print(response_data)
    if LOG_ENABLE:
        current_app.logger.info(f"response data in /{request_name}")
        current_app.logger.info(response_data)

@app.route("/freqDomainConversion", methods=["POST"])
def freq_domain_conversion():
    """
    时域信号转变为频域信号
    :return:
    """
    try:
        request_name = "freqDomainConversion"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        val_type = request_data["valType"]
        if val_type == "speed":
            vibration_data = signal_processor.cal_vibration_data(request_data["csvData"])
            time_domain_curve_raw = {}
            for axis in ["X", "Y", "Z"]:
                time_domain_curve_raw[axis] = vibration_data[axis]["time_domain"]["val"]

            if "windowType" in request_data:
                axis_time_freq = signal_processor.cal_axis_time_freq_domain_data_by_curve(
                    time_domain_curve_raw=time_domain_curve_raw,
                    unit=vibration_data["unit"],
                    samp_freq=vibration_data["sampling_freq"],
                    code_window_type=request_data["windowType"]
                )
            else:
                axis_time_freq = signal_processor.cal_axis_time_freq_domain_data_by_curve(
                    time_domain_curve_raw=time_domain_curve_raw,
                    unit=vibration_data["unit"],
                    samp_freq=vibration_data["sampling_freq"]
                )
        else:
            if "windowType" in request_data:
                axis_time_freq = signal_processor.cal_axis_time_freq_domain_data(
                    csv_data_str=request_data["csvData"],
                    code_window_type=request_data["windowType"]
                )
            else:
                axis_time_freq = signal_processor.cal_axis_time_freq_domain_data(csv_data_str=request_data["csvData"])

        signal_processor.show_signal(x=axis_time_freq["X"]["time_domain"]["time"], y=axis_time_freq["X"]["time_domain"]["val"])

        response_axis_time_freq = {"unit": axis_time_freq["unit"]}
        for axis in ["X", "Y", "Z"]:
            response_axis_time_freq[axis.lower()+"data"] = {
                "timeDomain": {
                    "x": list(axis_time_freq[axis]["time_domain"]["time"]),
                    # "y": list(axis_time_freq[axis]["time_domain"]["val"]),
                    "y": list(axis_time_freq[axis]["time_domain"]["normalized_val"]),
                },
                "freqDomain": {
                    "x": list(axis_time_freq[axis]["freq_domain"]["freq"]),
                    "y": list(axis_time_freq[axis]["freq_domain"]["amplitude"]),
                    "idxPeak": signal_processor.post_process_data(axis_time_freq[axis]["freq_domain"]["idx_peak"]),
                    "idxNearestPeak": signal_processor.post_process_data(axis_time_freq[axis]["freq_domain"]["idx_nearest_peak"]),
                }
            }
        response = {
            "status": CODE_SUCCESS,
            "data": response_axis_time_freq,
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/filter", methods=["POST"])
def cal_filtered_signal():
    """
    时域信号滤波
    :return:
    """
    try:
        request_name = "filter"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        cur_axis = request_data["axis"].upper()
        val_type = request_data["valType"]
        if val_type == "speed":
            vibration_data = signal_processor.cal_vibration_data(request_data["csvData"])
            cur_axis_time_data = {
                "sampling_freq": vibration_data["sampling_freq"],
                "time": vibration_data[cur_axis]["time_domain"]["time"],
                "val": vibration_data[cur_axis]["time_domain"]["val"],
                "normalized_val": vibration_data[cur_axis]["time_domain"]["normalized_val"],
            }
        else:
            cur_axis_time_data = signal_processor.cal_cur_axis_time_domain_data(csv_data_str=request_data["csvData"], axis=cur_axis)

        """
        filtered_data = signal_processor.signal_processor_filter(
            input_data=np.array(request_data["signalData"]), samp_freq=request_data["sampFreq"],
            cut_off_freq_low=request_data["cutOffFreqLow"], cut_off_freq_high=request_data["cutOffFreqHigh"],
            filter_func_code=request_data["filterFuncCode"]
        )
        """
        filtered_data = signal_processor.signal_processor_filter(
            input_data=np.array(cur_axis_time_data["normalized_val"]),
            samp_freq=cur_axis_time_data["sampling_freq"],
            cut_off_freq_low=request_data["cutOffFreqLow"],
            cut_off_freq_high=request_data["cutOffFreqHigh"],
            filter_func_code=request_data["filterFuncCode"]
        )

        response = {
            "status": CODE_SUCCESS,
            "filteredData": list(filtered_data),
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }

    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/calEnvelopeDemodulation", methods=["POST"])
def cal_envelope_demodulation():
    """
    根据传感器原始数据，计算包络解调结果
    :return:
    """
    try:
        request_name = "calEnvelopeDemodulation"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        cur_axis = request_data["axis"].upper()
        val_type = request_data["valType"]
        if val_type == "speed":
            vibration_data = signal_processor.cal_vibration_data(request_data["csvData"])

            if "windowType" in request_data:
                envelope_demodulation = signal_processor.cal_signal_envelope_demodulation(
                    signal=vibration_data[cur_axis]["time_domain"]["normalized_val"],
                    samp_freq=vibration_data["sampling_freq"],
                    cut_off_freq_low=request_data["cutOffFreqLow"], cut_off_freq_high=request_data["cutOffFreqHigh"],
                    code_window_type=request_data["windowType"]
                )
            else:
                envelope_demodulation = signal_processor.cal_signal_envelope_demodulation(
                    signal=vibration_data[cur_axis]["time_domain"]["normalized_val"],
                    samp_freq=vibration_data["sampling_freq"],
                    cut_off_freq_low=request_data["cutOffFreqLow"], cut_off_freq_high=request_data["cutOffFreqHigh"]
                )
            envelope_demodulation["unit"] = vibration_data["unit"]
        else:
            if "windowType" in request_data:
                envelope_demodulation = signal_processor.envelope_demodulate(
                    csv_data_str=request_data["csvData"], axis=request_data["axis"],
                    cut_off_freq_low=request_data["cutOffFreqLow"], cut_off_freq_high=request_data["cutOffFreqHigh"],
                    code_window_type=request_data["windowType"]
                )
            else:
                envelope_demodulation = signal_processor.envelope_demodulate(
                    csv_data_str=request_data["csvData"], axis=request_data["axis"],
                    cut_off_freq_low=request_data["cutOffFreqLow"], cut_off_freq_high=request_data["cutOffFreqHigh"]
                )

        response = {
            "status": CODE_SUCCESS,
            "data": {
                "freqDomain": {
                    "x": list(envelope_demodulation["freq_domain"]["freqs"]),
                    "y": list(envelope_demodulation["freq_domain"]["amplitude_response"]),
                    "idxPeak": signal_processor.post_process_data(envelope_demodulation["freq_domain"]["idx_peak"]),  # 防止只有单个值被转换为np.float64，导致无法被list()处理，故外部再做一次np.array()转换
                    "idxNearestPeak": signal_processor.post_process_data(envelope_demodulation["freq_domain"]["idx_nearest_peak"]),
                },
                "envelopeDemodulation": {
                    # "x": list(envelope_demodulation["envelope_demodulation"]["time"]),
                    # "y": list(envelope_demodulation["envelope_demodulation"]["val"]),
                    "x": list(envelope_demodulation["envelope_demodulation"]["freqs"]),
                    "y": list(envelope_demodulation["envelope_demodulation"]["envelope_spectrum"]),
                    "idxPeak": signal_processor.post_process_data(envelope_demodulation["envelope_demodulation"]["idx_peak"]),  # 防止只有单个值被转换为np.float64，导致无法被list()处理，故外部再做一次np.array()转换
                    "idxNearestPeak": signal_processor.post_process_data(envelope_demodulation["envelope_demodulation"]["idx_nearest_peak"]),
                },
                "unit": envelope_demodulation["unit"]
            }
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/calTimeDomainCurve", methods=["POST"])
def cal_time_domain_curve():
    """
    将传感器原始数据转换为时域信号
    :return:
    """
    try:
        request_name = "calTimeDomainCurve"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        val_type = request_data["valType"]
        if val_type == "speed":
            vibration_data = signal_processor.cal_vibration_data(request_data["csvData"])
            axis_time = {
                "unit": vibration_data["unit"]
            }
            for axis in ["X", "Y", "Z"]:
                axis_time[axis] = vibration_data[axis]["time_domain"]
        else:
            axis_time = signal_processor.cal_axis_time_domain_data(csv_data_str=request_data["csvData"])

        response_axis_time_freq = {
            "unit": axis_time["unit"]
        }
        for axis in ["X", "Y", "Z"]:
            response_axis_time_freq[axis.lower()+"data"] = {
                "x": list(axis_time[axis]["time"]),
                # "y": list(axis_time[axis]["val"]),
                "y": list(axis_time[axis]["normalized_val"]),
            }
        response = {
            "status": CODE_SUCCESS,
            "data": response_axis_time_freq,
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/calSpeedTimeDomainCurve", methods=["POST"])
def cal_speed_time_domain_curve():
    """
    将传感器原始加速度数据转换为时域的速度数据
    :return:
    """
    try:
        request_name = "calSpeedTimeDomainCurve"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        axis = request_data["axis"]
        vibration_data = signal_processor.cal_vibration_data(
            csv_data_str=request_data["csvData"],
            start_time=request_data["startTime"],
            len_ratio=request_data["lenRatio"]
        )

        signal_processor.show_signal(x=vibration_data[axis]["time_domain"]["time"], y=vibration_data[axis]["time_domain"]["val"])

        response_axis_speed_time = {
            "unit": vibration_data["unit"]
        }
        for axis in ["X", "Y", "Z"]:
            response_axis_speed_time[axis.lower() + "data"] = {
                "timeDomain": {
                    "x": list(vibration_data[axis]["time_domain"]["time"]),
                    "y": list(vibration_data[axis]["time_domain"]["val"]),
                    # "y": vibration_data[axis]["time_domain"]["normalized_val"],
                },
                # "freqDomain": {
                #     "x": vibration_data[axis]["freq_domain"]["freq"],
                #     "y": vibration_data[axis]["freq_domain"]["amplitude"],
                # }
            }
        response = {
            "status": CODE_SUCCESS,
            "data": response_axis_speed_time
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/calSignalEffectiveVal", methods=["POST"])
def cal_signal_effective_val():
    """
    计算信号有效值(rms)
    :return:
    """
    try:
        request_name = "calSignalEffectiveVal"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        val_type = request_data["valType"]
        effective_val_data = signal_processor.cal_signal_rms_val(csv_data_str=request_data["csvData"], val_type=val_type)

        response_effective_val_data = {
            "unit": effective_val_data["unit"],
            "effectiveVal": {},
        }
        for axis in ["X", "Y", "Z"]:
            response_effective_val_data["effectiveVal"][axis.lower()] = effective_val_data["rms"][axis]

        response = {
            "status": CODE_SUCCESS,
            "data": response_effective_val_data
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/calSignalPeakToPeakVal", methods=["POST"])
def cal_signal_peak_to_peak_val():
    """
    计算信号峰峰值
    :return:
    """
    try:
        request_name = "calSignalPeakToPeakVal"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        val_type = request_data["valType"]
        effective_val_data = signal_processor.cal_signal_p2p_val(csv_data_str=request_data["csvData"], val_type=val_type)

        response_peak_to_peak_val_data = {
            "unit": effective_val_data["unit"],
            "peakToPeakVal": {},
        }
        for axis in ["X", "Y", "Z"]:
            response_peak_to_peak_val_data["peakToPeakVal"][axis.lower()] = effective_val_data["p2p"][axis]

        response = {
            "status": CODE_SUCCESS,
            "data": response_peak_to_peak_val_data
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/calAudioSignalDecibel", methods=["POST"])
def cal_audio_signal_decibel():
    """
    计算声音信号分贝数
    :return:
    """
    try:
        request_name = "calAudioSignalDecibel"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        val_type = request_data["valType"]
        if val_type != "audio":
            raise Exception("数据类型必须为声音数据")
        audio_signal_decibel_data = signal_processor.cal_audio_signal_decibel_val(csv_data_str=request_data["csvData"])
        response = {
            "status": CODE_SUCCESS,
            "data": audio_signal_decibel_data
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/calSignalFeature", methods=["POST"])
def cal_signal_feature():
    """
    计算信号特征值，包括有效值、峭度指标、脉冲指标等
    :return:
    """
    try:
        request_name = "calSignalFeature"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        start_time = 0
        if "startTime" in request_data:
            start_time = request_data["startTime"]

        len_ratio = 1
        if "lenRatio" in request_data:
            len_ratio = request_data["lenRatio"]

        val_type = request_data["valType"]

        # 加入起始ratio和长度ratio参数，用于测试获取和现有分析接近的特征阐述所需的最小采样长度
        feature_val_data = signal_processor.cal_axis_signal_feature_data(
            csv_data_str=request_data["csvData"],
            val_type=val_type,
            start_time=start_time,
            len_ratio=len_ratio
        )

        response_feature_val_data = {
            "unit": feature_val_data["unit"]
        }
        for axis in ["X", "Y", "Z"]:
            response_feature_val_data[axis.lower() + "data"] = {
                "mean": response_float_num_trans(data=feature_val_data[axis]["mean"]),
                "max": response_float_num_trans(data=feature_val_data[axis]["max"]),
                "min": response_float_num_trans(data=feature_val_data[axis]["min"]),
                "effectiveVal": response_float_num_trans(data=feature_val_data[axis]["rms"]),
                "kurtosisVal": response_float_num_trans(data=feature_val_data[axis]["kurtosis_val"]),
                "skewnessVal": response_float_num_trans(data=feature_val_data[axis]["skewness_val"]),
                "marginFactor": response_float_num_trans(data=feature_val_data[axis]["margin_factor"]),
                "crestFactor": response_float_num_trans(data=feature_val_data[axis]["crest_factor"]),
                "shapeFactor": response_float_num_trans(data=feature_val_data[axis]["shape_factor"]),
                "impulseFactor": response_float_num_trans(data=feature_val_data[axis]["impulse_factor"]),
                "meanAmplitude": response_float_num_trans(data=feature_val_data[axis]["mean_amplitude"]),
                "rootSquareAmplitude": response_float_num_trans(data=feature_val_data[axis]["root_square_amplitude"]),
                "peakVal": response_float_num_trans(data=feature_val_data[axis]["peak_val"]),
                "peakToPeakVal": response_float_num_trans(data=feature_val_data[axis]["peak_to_peak_val"]),
            }
        response = {
            "status": CODE_SUCCESS,
            "data": response_feature_val_data
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/crossPhaseAnalyse", methods=["POST"])
def cal_cross_phase_analysis():
    """
    基于传感器原始数据计算交叉相位信息
    :return:
    """
    try:
        request_name = "crossPhaseAnalyse"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        cross_phase_analysis = signal_processor.cross_phase_analyse(csv_data_str=request_data["csvData"])

        response_cross_phase_analysis = {
            "x": cross_phase_analysis["freqs"],
            "complementaryAmplitudeXY": cross_phase_analysis["complementary_amplitude_response_xy"],
            "complementaryAmplitudeYZ": cross_phase_analysis["complementary_amplitude_response_yz"],
            "complementaryAmplitudeZX": cross_phase_analysis["complementary_amplitude_response_zx"],
            "phaseDiffXY": cross_phase_analysis["phase_response_diff_xy"],
            "phaseDiffYZ": cross_phase_analysis["phase_response_diff_yz"],
            "phaseDiffZX": cross_phase_analysis["phase_response_diff_zx"],
        }

        response = {
            "status": CODE_SUCCESS,
            "data": response_cross_phase_analysis,
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/detectCameraMove", methods=["POST"])
def detect_camera_move():
    """
    输入安装在摄像机上的振动传感器数据，返回摄像机是否被人移动的检测结果
    :return:
    """
    try:
        request_name = "detectCameraMove"
        request_data = pretreat_request_data(request=request, request_name=request_name)
        camera_move = signal_processor.diagnose_camera_move(csv_data_str=request_data["csvData"])
        response = {
            "status": CODE_SUCCESS,
            "cameraMove": camera_move,
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/diagnose", methods=["POST"])
def diagnose():
    """
    输入设备振动传感器数据，返回诊断结果
    :return:
    """
    try:
        request_name = "diagnose"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        if "suppData" in request_data:
            supp_data = request_data["suppData"]
        else:
            supp_data = None

        condition_code = signal_processor.diagnose_equipment(
            csv_data_str=request_data["csvData"],
            equip_type=request_data["equipType"],
            supp_data=supp_data
        )

        response = {
            "status": CODE_SUCCESS,
            "conditionCode": condition_code,
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/showSignalFeature", methods=["POST"])
def show_signal_feature():
    """
    本地测试用，基于传感器原始数据，生成目标数据的平方自谱、线性自谱、PSD功率密度谱、ESD能量密度谱，用plotly展示
    :return:
    """
    try:
        request_name = "showSignalFeature"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        # from signal_processor_bare import show_freq_analysis
        signal_processor.show_freq_analysis(csv_data_str1=request_data["csvDataLowAcc"], csv_data_str2=request_data["csvDataHighAcc"])

        response = {
            "status": CODE_SUCCESS,
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/calDataEnvelopeDemodulation", methods=["POST"])
def response_cal_data_envelope_demodulation():
    """
    本地测试用，计算目标轴信号的峰值包络解调的PSD
    :return:
    """
    try:
        request_name = "calFreqSpectrum"
        request_data = pretreat_request_data(request=request, request_name=request_name)

        if "windowType" in request_data:
            cur_axis_envelope_demodulation = signal_processor.cal_data_envelope_demodulation(
                csv_data_str=request_data["csvData"],
                axis=request_data["axis"],
                cut_off_freq_low=request_data["cutOffFreqLow"],
                cut_off_freq_high=request_data["cutOffFreqHigh"],
                code_envelope_cal_mode=request_data["envelopeCalMode"],
                code_window_type=request_data["windowType"]
            )
        else:
            cur_axis_envelope_demodulation = signal_processor.cal_data_envelope_demodulation(
                csv_data_str=request_data["csvData"],
                axis=request_data["axis"],
                cut_off_freq_low=request_data["cutOffFreqLow"],
                cut_off_freq_high=request_data["cutOffFreqHigh"],
                code_envelope_cal_mode=request_data["envelopeCalMode"],
            )
        response = {
            "status": CODE_SUCCESS,
            "data": {
                "freqDomain": {
                    "x": list(cur_axis_envelope_demodulation["freq_domain"]["freqs"]),
                    "y": list(cur_axis_envelope_demodulation["freq_domain"]["amplitude_response_psd"]),
                    # "idxPeak": signal_processor.post_process_data(envelope_demodulation["freq_domain"]["idx_peak"]),
                    # 防止只有单个值被转换为np.float64，导致无法被list()处理，故外部再做一次np.array()转换
                    # "idxNearestPeak": signal_processor.post_process_data(envelope_demodulation["freq_domain"]["idx_nearest_peak"]),
                },
                "envelopeDemodulation": {
                    "x": list(cur_axis_envelope_demodulation["envelope_demodulation"]["freqs"]),
                    "y": list(cur_axis_envelope_demodulation["envelope_demodulation"]["envelope_spectrum_psd"]),
                    # "idxPeak": signal_processor.post_process_data(envelope_demodulation["envelope_demodulation"]["idx_peak"]),
                    # 防止只有单个值被转换为np.float64，导致无法被list()处理，故外部再做一次np.array()转换
                    # "idxNearestPeak": signal_processor.post_process_data(envelope_demodulation["envelope_demodulation"]["idx_nearest_peak"]),
                },
                # "unit": envelope_demodulation["unit"]
            }
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

@app.route("/generateMachineCode", methods=["GET"])
def generate_machine_code():
    for k, v in request.headers.items():
        print(f"{k}: {v}")

    request_name = "generateMachineCode"
    try:
        code = signal_processor.generate_local_machine_code()
        response = {
            "status": CODE_SUCCESS,
            "code": code
        }
    except Exception as e:
        response = {
            "status": CODE_ERROR,
            "errorMessage": str(e),
        }
    pretreat_response_data(response_data=response, request_name=request_name)
    return jsonify(response)

#===================================================================================
@app.route("/analyzeewt", methods=["POST"])  # 修改路由避免冲突
# @swag_from({
#     'requestBody': {
#         'required': True,
#         'content': {
#             'application/json': {
#                 'schema': {
#                     'type': 'object',
#                     'required': ['csvData'],
#                     'properties': {
#                         'csvData': {'type': 'string','description': 'CSV 原始内容（字符串）'},
#                         'decayFactor': {'type': 'number', 'default': 0.5},
#                         'valType': {'type': 'string', 'description': 'signal type'},
#                         'serial_no': {'type': 'string', 'description': '数据序列编号'}
#                     }
#                 }
#             }
#         }
#     },
#     'responses': {200: {'description': '分析结果'}}
# })
def ewt_analyze():

    try:
        request_name = "analyzeewt"
        data = pretreat_request_data(request=request, request_name=request_name)
        # data = request.get_json()
        if not data or 'csvData' not in data:
            return jsonify({"error": "缺少csvData参数"}), 400

        # 处理请求参数
        csv_string = data['csvData'].replace('\r\n', '\n')
        decay_factor = float(data.get('decayFactor', 0.5))
        serial_no = data.get('serialNo', f"sn_{int(time.time())}")
        # timestamp = data.get('timestamp')


        # 数据库配置（根据实际情况修改）
        db_config = {
            "host": "47.116.67.106",
            "port": 3306,
            "user": "root",
            "password": "bangding123",
            "database": "pdm_data"
        }

        # 生成唯一ID
        unique_id = hashlib.md5((csv_string + str(time.time())).encode()).hexdigest()

        # 执行EWT分析
        processor = EWTProcessor(
            csv_string=csv_string,
            decay_factor=decay_factor,
            db_config=db_config,
            serial_no=serial_no
        )
        processor.analyze_all_axes()

        # 缓存处理结果
        with global_processor_lock:
            global_processor_map[unique_id] = processor
        analyze_cache[unique_id] = processor.results

        # 构造响应
        # return Response(
        #     json.dumps({
        #         "EWTresults": {axis: processor._format_axis_data(axis) for axis in ['X', 'Y', 'Z']},
        #         "SVDresults": {
        #             axis: {
        #                 "singular_values": np.round(processor.results[axis]['svd']['singular_values'], 2).tolist(),
        #                 "d_i": float(np.round(processor.results[axis]['svd']['d_i'], 2))
        #             } for axis in ['X', 'Y', 'Z']
        #         },
        #         "SigmaBars": {
        #             axis: np.round(processor.sigma_updater_map[axis].sigma_bar, 2).tolist()
        #             for axis in ['X', 'Y', 'Z']
        #         }
        #     }, cls=EWTProcessor.NumpyEncoder),
        #     content_type='application/json'
        # )
        return Response(
            json.dumps({
                "data": {
                    "dix": float(np.round(processor.results['X']['svd']['d_i'], 2)),
                    "diy": float(np.round(processor.results['Y']['svd']['d_i'], 2)),
                    "diz": float(np.round(processor.results['Z']['svd']['d_i'], 2))
                },
                # "timestamp": timestamp
                "status":0
            }),
            content_type='application/json'
        )

    except Exception as e:
        logging.error(f"EWT分析失败: {str(e)}")
        return Response(
            json.dumps({
                "code": 1,
                "message": "分析失败",
                "data": None,
                "error": str(e)
            }),
            content_type='application/json'
        ), 500
    

#===================================================================================

if __name__ == "__main__":
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    run_app(app=app, port=SERVE_PORT)