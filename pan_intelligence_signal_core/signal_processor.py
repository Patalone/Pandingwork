# -*- coding: utf-8 -*-
# ---------------------------------------------------------------
# author： Ash
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
import numpy as np
import pandas as pd
from io import StringIO
import math
import collections
import datetime
from datetime import timezone
# from intersect import intersection
import scipy
import scipy.signal as ss
from scipy.signal import hilbert, welch, find_peaks, csd
from scipy.interpolate import CubicSpline
from scipy.fftpack import fft, ifft
from scipy.stats import kurtosis, skew

import time
import sys
import os
import platform
import subprocess
# import h5py
import hashlib
from io import BytesIO
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ALLOW_SHOW_FIGURE = True

GRAVITY_ACC = 9.80665
P_REF = 20 * 1e-6  # 计算分贝的标准参考声压值为20微帕

CODE_FILTER_BOND_PASS = 0  # 带通滤波功能标识码
CODE_FILTER_BOND_STOP = 1  # 带阻滤波功能标识码
CODE_FILTER_HIGH_PASS = 2  # 高通滤波功能标识码
CODE_FILTER_LOW_PASS = 3  # 低通滤波功能标识码

FILTER_WN_CUT_LIMIT = 3.3e-1  # 避免滤波的高通截止频率过小导致滤波后的信号数值发散的问题

TOP_PEAK_LEN_DEFAULT = 6

CODE_RAW_DATA_AXIS_X = "X"
CODE_RAW_DATA_AXIS_Y = "Y"
CODE_RAW_DATA_AXIS_Z = "Z"
CODE_AXIS_X = "X"
CODE_AXIS_Y = "Y"
CODE_AXIS_Z = "Z"

CODE_WINDOW_TYPE_HAMMING = "hamming"
CODE_WINDOW_TYPE_HANNING = "hanning"
CODE_WINDOW_TYPE_WAVELET = "wavelet"
CODE_WINDOW_TYPE_DEFAULT = CODE_WINDOW_TYPE_HANNING

CODE_ENVELOPE_CAL_MODE_HILBERT = "hilbert"
CODE_ENVELOPE_CAL_MODE_PEAK_CONECT = "peak_connect"

CODE_EQUIP_TYPE_MOTOR = "motor"
CODE_EQUIP_TYPE_BEARING = "bearing"
CODE_EQUIP_TYPE_CRANKSHAFT = "crankshaft"
CODE_EQUIP_TYPE_GEARBOX = "gearbox"
CODE_EQUIP_TYPE_BEAM_CENTER_AXIS = "beam_center_axis"
CODE_EQUIP_TYPE_FOOTING = "footing"

CODE_VAL_TYPE_ACC_LOW = "lowAcc"
CODE_VAL_TYPE_ACC_HIGH = "highAcc"
CODE_VAL_TYPE_VIBRATION = "speed"

THRESHOLD_NOISE_STATUS_UP_NORMAL = 100
THRESHOLD_NOISE_STATUS_UP_ABNORMAL_LOW = 105
THRESHOLD_NOISE_STATUS_UP_ABNORMAL_MID = 110

THRESHOLD_BEARING_STATUS_UP_NORMAL = 3
THRESHOLD_BEARING_STATUS_UP_ABNORMAL_LOW = 5
THRESHOLD_BEARING_STATUS_UP_ABNORMAL_MID = 9

THRESHOLD_ROT_SPD_UP_SLOW = 500
THRESHOLD_ROT_SPD_UP_MID = 1800

CODE_CONDITION_NORMAL = "normal"
CODE_CONDITION_BEARING_WEAR = "bearing_wear"
CODE_CONDITION_BEARING_ABNORMAL_LOW = "bearing_risk_60_day"
CODE_CONDITION_BEARING_ABNORMAL_MID = "bearing_risk_30_day"
CODE_CONDITION_BEARING_ABNORMAL_HIGH = "bearing_risk_15_day"
CODE_CONDITION_NOISE_ABNORMAL_LOW = "noise_abnormal_low"
CODE_CONDITION_NOISE_ABNORMAL_MID = "noise_abnormal_mid"
CODE_CONDITION_NOISE_ABNORMAL_HIGH = "noise_abnormal_high"

ROD_SPD_ACTUAL_INACCURATE = False  # 用于表征现有的从系统中取出的转速数据是否准确。目前转速不准确导致阈值报警过多，用此值标记，暂时跳过阈值报警阶段，只使用轴承状态来诊断

DICT_RISK_STATUS = {
    CODE_CONDITION_NORMAL: 0,
    CODE_CONDITION_BEARING_ABNORMAL_LOW: 1,
    CODE_CONDITION_BEARING_ABNORMAL_MID: 2,
    CODE_CONDITION_BEARING_ABNORMAL_HIGH: 3
}

DEFAULT_CSV_FILE_DELIMITER = ";"

N_DECIMAL_PLACE = 5

CSV_HEADER_INFO = {
    "version_row_idx": 0,
    "data_collect_unit_row_idx": 1,
    "sensor_type_row_idx": 2,
    "timestamp_row_idx": 3,
    "sampling_freq_row_idx": 4,
    "unit_after_scaling_row_idx": 5,
    "scale_data_row_idx": 6,
    "title_row_idx": 9
}

DIC_FILTER_FUNC = {
    CODE_FILTER_BOND_PASS: "bandpass",
    CODE_FILTER_BOND_STOP: "bandstop",
    CODE_FILTER_HIGH_PASS: "highpass",
    CODE_FILTER_LOW_PASS: "lowpass",
}

SPEED_CURVE_START_TIME = np.float64(1.8)
SPEED_CURVE_LEN_RATIO = np.float64(0.5)

ABS_FILE = os.path.abspath(__file__)
ABS_PATH = os.path.dirname(os.path.dirname(ABS_FILE))
ABS_CONFIG_PATH = os.path.join(ABS_PATH, "config.yml")
ABS_LICENSE_PATH = os.path.join(ABS_PATH, "license.bangding")
# ABS_SUPERVISORD_PATH = os.path.join(ABS_PATH, "supervisord_temp.conf")
# ABS_NSSM_PATH = os.path.join(ABS_PATH, "nssm.exe")
ABS_PYTHON_PATH = sys.executable

def handle_nan(data: np.float64) -> np.float64:
    if np.isnan(data):
        return None
    else:
        return data

def get_code_axis(axis: str) -> str:
    if axis.upper() == "X":
        return CODE_AXIS_X
    elif axis.upper() == "Y":
        return CODE_AXIS_Y
    elif axis.upper() == "Z":
        return CODE_AXIS_Z
    else:
        raise(Exception("轴编号不正确，请检查"))

def init_signal_processor():
    _MESSAGE_ERR_NOT_VALID = "服务未取得授权或已失效，请联系供应商取得授权或延长服务时效"
    global _valid_date
    _valid_date = None

    def _get_cur_timestamp():
        timestamp = int(time.mktime(time.localtime(time.time())))
        return timestamp

    def _create_basic_core():
        class _Basic_Core:
            def __init__(self, key: bytes, hash_key: str) -> None:
                self.__key = key
                self.__hash_key = hash_key
                # Choose a valid AES key size (128, 192, or 256 bits)
                self.__VALID_KEY_SIZE = 256
                self.__adjusted_key = self.__adjust_key_size(key)
                # self.__PUBLIC_KEY_DATA = b"0Y0\x13\x06\x07*\x86H\xce=\x02\x01\x06\x08*\x86H\xce=\x03\x01\x07\x03B\x00\x04\x18\xe4\x0fx\x93\xaaUS\x90\x94\xb0OA\x97\x8e\xc7\x86\xf6]$\xb50Pmsq\xba\xcfh\xeeNs\xe0\t\xce,\x9ceP\xa3A\xeco\xcd\xec\xe8xc\x18+l\x97\xbe\x8a\xea\xcf\x9d\xe2\xa7`\x80n\x9a\x98"
                # self.__PUBLIC_KEY_DATA_LEN = len(self.__PUBLIC_KEY_DATA)
                self.__SIGNATURE_LEN = 128
                # self.__HASH_ADD_FIX_KEY = "z9xtv3r3xdm5kzjr763x0o353rzxbv3b"
                self.__MACHINE_CODE_LEN = 16
                self.__VALID_TIME_CODE_LEN = 8
                self.__LICENSE_VALID_BEFORE_TIMESTAMP = 0

            def __adjust_key_size(self, key: bytes) -> bytes:
                # Truncate or adjust the key size
                adjusted_key = key[:self.__VALID_KEY_SIZE // 8]
                return adjusted_key

            def __remove_pad_signature(self, padded_signature: bytes) -> bytes:
                signature = padded_signature.rstrip(b"\x00")
                return signature

            def __encrypt_bytes(self, text: bytes) -> bytes:
                iv = os.urandom(16)  # Initialization vector
                cipher = Cipher(algorithms.AES(self.__adjusted_key), modes.CFB(iv), backend=default_backend())
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(text) + encryptor.finalize()
                return iv + ciphertext

            def __decrypt_bytes(self, encrypted_text: bytes) -> bytes:
                iv = encrypted_text[:16]
                ciphertext = encrypted_text[16:]
                cipher = Cipher(algorithms.AES(self.__adjusted_key), modes.CFB(iv), backend=default_backend())
                decryptor = cipher.decryptor()
                decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
                return decrypted_bytes

            def encrypt_str(self, input_str: str) -> str:
                text = str.encode(input_str, encoding="utf-8")
                encrypted_text = self.__encrypt_bytes(text=text)
                encrypted_str = encrypted_text.hex()
                return encrypted_str

            def decrypt_bytes_to_str(self, encrypted_text: bytes, mode: str = None) -> str:
                decrypt_text = self.__decrypt_bytes(encrypted_text=encrypted_text)
                decrypt_str = decrypt_text.decode(encoding="utf-8")
                if mode == "machine_encrypted_code":
                    decrypt_str = decrypt_str[:-self.__VALID_TIME_CODE_LEN]
                return decrypt_str

            def decrypt_file(self, input_file: str) -> bytes:
                encrypted_text = self.verify_signature(file_path=input_file)
                if encrypted_text is not None:
                    file_text = self.__decrypt_bytes(encrypted_text=encrypted_text)
                else:
                    file_text = None
                return file_text

            def decrypt_and_load_file(self, input_file: str):
                file_text = self.decrypt_file(input_file=input_file)
                file_text_io = BytesIO(file_text)
                return file_text_io

            """
            def decrypt_and_load_model(self, input_file: str):
                model_text = self.decrypt_file(input_file=input_file)
                model_text_io = BytesIO(model_text)
                with h5py.File(model_text_io, "r") as h5f:
                    decrypted_model = load_model(h5f)
                return decrypted_model
            """

            def verify_signature(self, file_path: str) -> bytes:
                with open(file_path, "rb") as file:
                    file_text = file.read()

                public_key_data = _get_pub_key()
                public_key_data_len = len(public_key_data)

                # file_public_key_data = file_text[:self.__PUBLIC_KEY_DATA_LEN]
                file_public_key_data = file_text[:public_key_data_len]

                # if file_public_key_data != self.__PUBLIC_KEY_DATA:
                if file_public_key_data != public_key_data:
                    return None
                public_key = serialization.load_der_public_key(file_public_key_data, backend=default_backend())
                # text = file_text[self.__PUBLIC_KEY_DATA_LEN:-self.__SIGNATURE_LEN]
                text = file_text[public_key_data_len:-self.__SIGNATURE_LEN]

                padded_signature = file_text[-self.__SIGNATURE_LEN:]
                signature = self.__remove_pad_signature(padded_signature)

                # Create a hash of the file content
                digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
                digest.update(text)
                hash_value = digest.finalize()

                # Verify the signature using the public key
                try:
                    public_key.verify(
                        signature,
                        hash_value,
                        ec.ECDSA(utils.Prehashed(hashes.SHA256()))
                    )
                    return text
                except Exception as e:
                    # print("Signature verification failed:", str(e))
                    return None

            def get_local_machine_code(self, local_machine_raw_code: str) -> str:
                # 进行哈希计算
                hash_object = hashlib.sha256(local_machine_raw_code.encode())

                # 转换成16进制字符串
                hex_digest = hash_object.hexdigest()

                hex_digest = hex_digest + self.__hash_key
                hash_object = hashlib.sha256(hex_digest.encode())
                hex_digest = hash_object.hexdigest()

                code = hex_digest[:self.__MACHINE_CODE_LEN]
                return code

            def cal_license_valid_before_timestamp(self, license_str: str) -> int:
                license_valid_before_timestamp_hex = "0x" + license_str[-self.__VALID_TIME_CODE_LEN:]
                license_valid_before_timestamp = int(license_valid_before_timestamp_hex, 16)
                return license_valid_before_timestamp

        return _Basic_Core

    def _get_local_machine_raw_code() -> str:
        try:
            local_machine_origin_code = os.environ.get("BASE_BD_CODE")
            if local_machine_origin_code:
                return local_machine_origin_code
        except Exception as e:
            local_machine_origin_code = None

        system = platform.system()
        if system == "Windows":
            # Windows系统下获取主板序列号的命令
            # command = "wmic baseboard get serialnumber"
            command = "powershell -Command \"Get-WmiObject -Class Win32_BaseBoard | Select-Object SerialNumber\""
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            if result.returncode == 0:
                serial_number = result.stdout.strip().split('\n')[-1].strip()
                local_machine_origin_code = serial_number
            else:
                print(f"Failed to retrieve serial number: {result.stderr.strip()}")
                local_machine_origin_code = None
        elif system == "Linux":
            """
            import distro
            os_name = distro.name()
            if "Ubuntu" in os_name:
                command = "dmidecode -t 2 | grep Serial"
            elif "CentOS" in os_name:
                command = "dmidecode | grep -i 'serial number'"
            """
            command = "dmidecode -t 2 | grep Serial"
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            if result.returncode == 0 and len(result.stdout.strip()) > 0:
                serial_number = result.stdout.strip().split(':')[-1].strip()
                local_machine_origin_code = serial_number
            else:
                command = "dmidecode | grep -i 'serial number'"
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
                if result.returncode == 0:
                    serial_number = result.stdout.strip().split(':')[-1].strip()
                    local_machine_origin_code = serial_number
                else:
                    command = "lshw -class system | grep 'serial'"
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
                    if result.returncode == 0:
                        serial_number = result.stdout.strip().split(':')[-1].strip()
                        local_machine_origin_code = serial_number
                    else:
                        print(f"Failed to retrieve serial number: {result.stderr.strip()}")
                        local_machine_origin_code = None
        else:
            print(f"Unsupported operating system: {system}")
            local_machine_origin_code = None
        return local_machine_origin_code

    def _check_valid(encrypt_key: bytes, hash_key: str) -> bool:
        global _valid_date
        check_date = datetime.date.today()
        # check_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        # if _valid_date == check_time:
        if _valid_date == check_date:
            return True
        try:
            __basic_core_creator = _create_basic_core()
            __basic_core = __basic_core_creator(key=encrypt_key, hash_key=hash_key)
            license_text = __basic_core.verify_signature(file_path=ABS_LICENSE_PATH)
            if license_text is not None:
                license_str = __basic_core.decrypt_bytes_to_str(encrypted_text=license_text)
                license_machine_encrypted_code = __basic_core.decrypt_bytes_to_str(encrypted_text=license_text,
                                                                                   mode="machine_encrypted_code")
                license_machine_encrypted_code_text = bytes.fromhex(license_machine_encrypted_code)
                license_machine_code = __basic_core.decrypt_bytes_to_str(
                    encrypted_text=license_machine_encrypted_code_text)
                local_machine_code = __basic_core.get_local_machine_code(_get_local_machine_raw_code())

                if license_machine_code == local_machine_code:
                    license_valid_before_timestamp = __basic_core.cal_license_valid_before_timestamp(
                        license_str=license_str)
                    cur_timestamp = _get_cur_timestamp()
                    if cur_timestamp <= license_valid_before_timestamp:
                        _valid_date = check_date
                        # _valid_date = check_time
                        return True
        except Exception as e:
            # print(f"Error checking license: {e}")
            print(_MESSAGE_ERR_NOT_VALID)
            return False
        print(_MESSAGE_ERR_NOT_VALID)
        return False

    def _get_pub_key():
        __PUBLIC_KEY_DATA = b'0Y0\x13\x06\x07*\x86H\xce=\x02\x01\x06\x08*\x86H\xce=\x03\x01\x07\x03B\x00\x04\xd6\x82\x10\n\xde\x02:B\xff\xf3y\x87;\xaa\xbc\x1d\xa3\xab\xa0\xd6\xf5|!\x99\xe5\x19}V\xc1\xc0\xaa\xdeC\xb6H6VO:\xbaQ\xfdDt\xd0\x04k\x18iw\xae\x92F\x03\xc3\xc4\xe6C\\\xfd$:G\xc1'

        def __get_key():
            return __PUBLIC_KEY_DATA

        return __get_key()

    def _get_enc_key():
        __ENCRYPTION_KEY = b"f5d8c455385f7ebca26a139fb87236fb28fe2a7c6deddaa24d58fed701619aabf71ce1f0b7517ada0833b757d46137ddd041f147a076267e3a36aa9095b94c23"

        def __get_key():
            return __ENCRYPTION_KEY

        return __get_key()

    def _get_hash_add_key():
        __HASH_ADD_FIX_KEY = "k5efk01yu0lc1ba7q1xaaphooq4vem2d"

        def __get_key():
            return __HASH_ADD_FIX_KEY

        return __get_key()

    def _generate_local_machine_code() -> str:
        __basic_core_creator = _create_basic_core()
        __basic_core = __basic_core_creator(key=_get_enc_key(), hash_key=_get_hash_add_key())
        local_machine_origin_code = __basic_core.get_local_machine_code(_get_local_machine_raw_code())
        return __basic_core.encrypt_str(input_str=local_machine_origin_code)

    def keep_specified_decimal(input_array: np.array, n_decimal_place: np.int64 = N_DECIMAL_PLACE) -> np.array:
        return np.round(a=input_array, decimals=n_decimal_place)

    class __Signal_Processor:
        def __check(func):
            def wrapper(self, *args, **kwargs):
                valid = _check_valid(encrypt_key=_get_enc_key(), hash_key=_get_hash_add_key())
                if not valid:
                    raise RuntimeError(_MESSAGE_ERR_NOT_VALID)
                else:
                    return func(self, *args, **kwargs)

            return wrapper

        def __init__(self):
            return None

        def generate_local_machine_code(self) -> str:
            return _generate_local_machine_code()

        def post_process_data(self, data: np.array) -> list:
            """
            将python计算结果转换为float64和list等可以被序列化作为请求相应数据的形式
            :param data: 需要后处理的数据
            :return: 可以序列化的float64的list数据
            """
            return list(np.array(np.float64(data)).flatten())

        
        def parse_data(self, csv_data_str: str = None, delimiter: str = DEFAULT_CSV_FILE_DELIMITER) -> dict:
            """
            解析csv文件内容字符串
            :param csv_data_str: csv文件内容字符串
            :return: 解析结果组成的字典对象
            """
            print("==================== CSV_HEADER_INFO 配置 ====================")
            print(CSV_HEADER_INFO)
            print("============================================================")
            if csv_data_str is None:
                raise(Exception("原始数据流/数据路径不能为空"))
            
            if len(csv_data_str) > 0:
                if os.path.exists(csv_data_str):  # 若传入csv_data_str为路径，则作为路径导入，否则作为字符数据流导入
                    csv_data_io = csv_data_str
                else:
                    csv_data_io = StringIO(csv_data_str)
            else:
                raise(Exception("原始数据读取出现错误，请检查原始数据流/数据路径"))

            # 读取CSV数据的前几行以获取配置信息
            num_config_rows = CSV_HEADER_INFO["title_row_idx"]
            config_data = pd.read_csv(csv_data_io, delimiter=delimiter, nrows=num_config_rows, header=None)

            print("==================== 通过 nrows 读取到的配置数据 (config_data) ====================")
            print(config_data)
            print("================================================================================")

            version = config_data.iloc[CSV_HEADER_INFO["version_row_idx"], 1]
            data_collect_unit = config_data.iloc[CSV_HEADER_INFO["data_collect_unit_row_idx"], 1]
            sensor_type = config_data.iloc[CSV_HEADER_INFO["sensor_type_row_idx"], 1]
            utc_time_str = config_data.iloc[CSV_HEADER_INFO["timestamp_row_idx"], 1]
            utc_datetime = datetime.datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%S")
            timestamp = utc_datetime.replace(tzinfo=timezone.utc).timestamp()
            sampling_freq = np.float64(config_data.iloc[CSV_HEADER_INFO["sampling_freq_row_idx"], 1])
            unit_after_scaling = config_data.iloc[CSV_HEADER_INFO["unit_after_scaling_row_idx"], 1]

            # 使用pandas读取CSV数据，跳过前几行
            if not os.path.exists(csv_data_str):
                csv_data_io.seek(0)  # 若csv_data_str为字符数据流，则重置 StringIO 的位置
            df = pd.read_csv(csv_data_io, delimiter=delimiter, header=CSV_HEADER_INFO["title_row_idx"])
            time_domain_curve_raw_X = np.array(df[CODE_RAW_DATA_AXIS_X].values)
            time_domain_curve_raw_Y = np.array(df[CODE_RAW_DATA_AXIS_Y].values)
            time_domain_curve_raw_Z = np.array(df[CODE_RAW_DATA_AXIS_Z].values)

            scale_data_row_val = np.float64(config_data.iloc[CSV_HEADER_INFO["scale_data_row_idx"], :])
            scale_data_row_val_valid = scale_data_row_val[np.isnan(scale_data_row_val) == False]
            scale_data_row_valid_cnt = len(scale_data_row_val_valid)
            if scale_data_row_valid_cnt == 1:
                scale_data = {
                    CODE_AXIS_X: scale_data_row_val_valid[0],
                    CODE_AXIS_Y: scale_data_row_val_valid[0],
                    CODE_AXIS_Z: scale_data_row_val_valid[0],
                }
            elif scale_data_row_valid_cnt == 3:
                scale_data = {
                    CODE_AXIS_X: scale_data_row_val_valid[0],
                    CODE_AXIS_Y: scale_data_row_val_valid[1],
                    CODE_AXIS_Z: scale_data_row_val_valid[2],
                }
            else:
                if np.sum(time_domain_curve_raw_X) == 0:
                    scale_data = {
                        CODE_AXIS_X: 1,
                        CODE_AXIS_Y: scale_data_row_val_valid[0],
                        CODE_AXIS_Z: scale_data_row_val_valid[1],
                    }
                elif np.sum(time_domain_curve_raw_Y) == 0:
                    scale_data = {
                        CODE_AXIS_X: scale_data_row_val_valid[0],
                        CODE_AXIS_Y: 1,
                        "Z": scale_data_row_val_valid[1],
                    }
                elif np.sum(time_domain_curve_raw_Z) == 0:
                    scale_data = {
                        CODE_AXIS_X: scale_data_row_val_valid[0],
                        CODE_AXIS_Y: scale_data_row_val_valid[1],
                        CODE_AXIS_Z: 1,
                    }
                else:
                    scale_data = {
                        CODE_AXIS_X: scale_data_row_val_valid[0],
                        CODE_AXIS_Y: scale_data_row_val_valid[1],
                        CODE_AXIS_Z: scale_data_row_val_valid[1],
                    }

            parsed_data = {
                "version": version,
                "data_collect_unit": data_collect_unit,
                "sensor_type": sensor_type,
                "utc_time_str": utc_time_str,
                "timestamp": timestamp,
                "sampling_freq": sampling_freq,
                "unit_after_scaling": unit_after_scaling,
                "scale_data": scale_data,
                "time_domain_curve_raw_X": time_domain_curve_raw_X,
                "time_domain_curve_raw_Y": time_domain_curve_raw_Y,
                "time_domain_curve_raw_Z": time_domain_curve_raw_Z,
            }
            return parsed_data

        def parse_data_from_csv_file(self, csv_file_path: str, delimiter: str = DEFAULT_CSV_FILE_DELIMITER) -> dict:
            """
            解析csv文件内容字符串
            :param csv_file_path: csv文件地址
            :return: 解析结果组成的字典对象
            """
            with open(csv_file_path, "r", encoding="utf-8") as file:
                csv_data_str = file.read()

            csv_data_io = StringIO(csv_data_str)

            # 读取CSV数据的前几行以获取配置信息
            num_config_rows = CSV_HEADER_INFO["title_row_idx"]
            config_data = pd.read_csv(csv_file_path, delimiter=delimiter, nrows=num_config_rows, header=None)

            version = config_data.iloc[CSV_HEADER_INFO["version_row_idx"], 1]
            data_collect_unit = config_data.iloc[CSV_HEADER_INFO["data_collect_unit_row_idx"], 1]
            sensor_type = config_data.iloc[CSV_HEADER_INFO["sensor_type_row_idx"], 1]
            utc_time_str = config_data.iloc[CSV_HEADER_INFO["timestamp_row_idx"], 1]
            utc_datetime = datetime.datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%S")
            timestamp = utc_datetime.replace(tzinfo=timezone.utc).timestamp()
            sampling_freq = np.float64(config_data.iloc[CSV_HEADER_INFO["sampling_freq_row_idx"], 1])
            unit_after_scaling = config_data.iloc[CSV_HEADER_INFO["unit_after_scaling_row_idx"], 1]

            # 使用pandas读取CSV数据，跳过前几行
            csv_data_io.seek(0)  # 重置 StringIO 的位置
            df = pd.read_csv(csv_data_io, delimiter=delimiter, header=CSV_HEADER_INFO["title_row_idx"])
            time_domain_curve_raw_X = np.array(df[CODE_RAW_DATA_AXIS_X].values)
            time_domain_curve_raw_Y = np.array(df[CODE_RAW_DATA_AXIS_Y].values)
            time_domain_curve_raw_Z = np.array(df[CODE_RAW_DATA_AXIS_Z].values)

            scale_data_row_val = np.float64(config_data.iloc[CSV_HEADER_INFO["scale_data_row_idx"], :])
            scale_data_row_val_valid = scale_data_row_val[np.isnan(scale_data_row_val) == False]
            scale_data_row_valid_cnt = len(scale_data_row_val_valid)
            if scale_data_row_valid_cnt == 1:
                scale_data = {
                    CODE_AXIS_X: scale_data_row_val_valid[0],
                    CODE_AXIS_Y: scale_data_row_val_valid[0],
                    CODE_AXIS_Z: scale_data_row_val_valid[0],
                }
            elif scale_data_row_valid_cnt == 3:
                scale_data = {
                    CODE_AXIS_X: scale_data_row_val_valid[0],
                    CODE_AXIS_Y: scale_data_row_val_valid[1],
                    CODE_AXIS_Z: scale_data_row_val_valid[2],
                }
            else:
                if np.sum(time_domain_curve_raw_X) == 0:
                    scale_data = {
                        CODE_AXIS_X: 1,
                        CODE_AXIS_Y: scale_data_row_val_valid[0],
                        CODE_AXIS_Z: scale_data_row_val_valid[1],
                    }
                elif np.sum(time_domain_curve_raw_Y) == 0:
                    scale_data = {
                        CODE_AXIS_X: scale_data_row_val_valid[0],
                        CODE_AXIS_Y: 1,
                        CODE_AXIS_Z: scale_data_row_val_valid[1],
                    }
                elif np.sum(time_domain_curve_raw_Z) == 0:
                    scale_data = {
                        CODE_AXIS_X: scale_data_row_val_valid[0],
                        CODE_AXIS_Y: scale_data_row_val_valid[1],
                        CODE_AXIS_Z: 1,
                    }
                else:
                    scale_data = {
                        CODE_AXIS_X: scale_data_row_val_valid[0],
                        CODE_AXIS_Y: scale_data_row_val_valid[1],
                        CODE_AXIS_Z: scale_data_row_val_valid[1],
                    }

            parsed_data = {
                "version": version,
                "data_collect_unit": data_collect_unit,
                "sensor_type": sensor_type,
                "utc_time_str": utc_time_str,
                "timestamp": timestamp,
                "sampling_freq": sampling_freq,
                "unit_after_scaling": unit_after_scaling,
                "scale_data": scale_data,
                "time_domain_curve_raw_X": time_domain_curve_raw_X,
                "time_domain_curve_raw_Y": time_domain_curve_raw_Y,
                "time_domain_curve_raw_Z": time_domain_curve_raw_Z,
            }
            return parsed_data

        def converse_unit(self, curve_data: np.array, scale_val: np.float64, unit_after_scaling: str) -> np.array:
            """
            曲线单位换算
            :param curve_data: 传感器获取的曲线原始值
            :param scale_val: 尺度缩放参数
            :param unit_after_scaling: 缩放后的数据的单位。如果为g，则需要再转换为m/s^2
            :return: 单位换算后的曲线数据
            """
            conversed_curve_data = curve_data * scale_val
            """
            if unit_after_scaling == "g":
                conversed_curve_data *= GRAVITY_ACC
            """
            # 修改，直接按照原始传感器数采单位输出
            return conversed_curve_data

        def cal_idx_peak(self, data: np.array) -> np.array:
            """
            输入原始数据，计算峰值索引
            :param data: 需要计算峰值索引的原始数据
            :return: 峰值索引数组
            """
            idx_peak, _ = find_peaks(data)
            return idx_peak

        def cal_t_array(self, n: np.int64, samp_freq: np.float64) -> np.array:
            """
            输入时域曲线采样点数、采样频率，输出时间轴数据
            :param n: 采样点数据
            :param samp_freq: 采样频率
            :return:
            """
            cycle_duration = 1 / samp_freq
            t_array = np.arange(0, n) * cycle_duration
            return t_array

        def adjust_curve(self, curve_raw: np.array, samp_freq: np.float64, start_time: np.float64,
                         len_ratio: np.float64) -> np.array:
            """
            调整信号
            :param curve_raw:  原始信号数组
            :param samp_freq:  采样频率
            :param start_time:  截取开始时间
            :param len_ratio:  截取长度占原数组长度的比例
            :return:
            """
            select_idx_start = np.int64(np.floor(start_time * samp_freq))
            n = len(curve_raw)
            select_len = np.round(n * len_ratio)
            select_idx_end = np.int64(np.min([select_idx_start + select_len, n]))
            # curve_adjust = curve_raw - np.mean(curve_raw)
            curve_adjust = curve_raw
            curve_adjust = curve_adjust[select_idx_start:select_idx_end]
            return curve_adjust

        def cal_axis_time_freq_domain_data_by_curve(self, time_domain_curve_raw: dict, unit: str, samp_freq: np.float64,
                                                    scale_val: np.float64=1, code_window_type: str=CODE_WINDOW_TYPE_DEFAULT) -> dict:
            """
            输入曲线数据字典对象，输出该数据XYZ轴的时域曲线和频域曲线
            :param time_domain_curve_raw:  原始的三轴时域数值数据
            :param unit:  时域数据单位
            :param samp_freq:  采样频率
            :param scale_val:  缩放系数（为了与csv字符串的处理方式保持一致）
            :return:
            """
            axis_time_freq_domain_data = {
                "unit": unit
            }
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                cur_axis_time_domain_curve_raw = time_domain_curve_raw[axis]
                axis_time_freq_domain_data[axis] = self.time_freq_data_process(
                    time_domain_curve_raw=cur_axis_time_domain_curve_raw,
                    scale_val=scale_val, unit_after_scaling=unit,
                    samp_freq=samp_freq, code_window_type=code_window_type
                )
            return axis_time_freq_domain_data

        def cal_axis_time_freq_domain_data(self, csv_data_str: str, code_window_type: str=CODE_WINDOW_TYPE_DEFAULT) -> dict:
            """
            输入csv字符串，输出该数据XYZ轴的时域曲线和频域曲线
            :param csv_data_str: csv文件内容字符串
            :return: 该csv字符串的三轴时域曲线和频域曲线
            """
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            axis_time_freq_domain_data = {
                "unit": csv_parsed_data["unit_after_scaling"]
            }
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                axis_time_freq_domain_data[axis] = self.time_freq_data_process(
                    time_domain_curve_raw=csv_parsed_data[f"time_domain_curve_raw_{axis}"],
                    scale_val=csv_parsed_data["scale_data"][axis],
                    unit_after_scaling=csv_parsed_data["unit_after_scaling"],
                    samp_freq=csv_parsed_data["sampling_freq"],
                    code_window_type=code_window_type
                )
            return axis_time_freq_domain_data

        def cal_cur_axis_time_domain_data(self, csv_data_str: str, axis: str) -> dict:
            """
            输入csv字符串和目标轴，输出该轴的时域数据曲线
            :param csv_data_str: csv文件内容字符串
            :param axis: 目标轴(X/Y/Z)
            :return: 该csv字符串的目标轴时域曲线
            """
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            cur_axis_time_domain_data = self.cal_time_data(
                time_domain_curve_raw=csv_parsed_data[f"time_domain_curve_raw_{axis}"],
                scale_val=csv_parsed_data["scale_data"][axis],
                unit_after_scaling=csv_parsed_data["unit_after_scaling"],
                samp_freq=csv_parsed_data["sampling_freq"]
            )
            cur_axis_time_domain_data_add_param = {
                "sampling_freq": csv_parsed_data["sampling_freq"],
                "time": cur_axis_time_domain_data["time"],
                "val": cur_axis_time_domain_data["val"],
                "normalized_val": cur_axis_time_domain_data["normalized_val"],
            }
            return cur_axis_time_domain_data_add_param

        def cal_axis_time_domain_data(self, csv_data_str: str) -> dict:
            """
            输入csv字符串，输出该数据XYZ轴的时域曲线
            :param csv_data_str: csv文件内容字符串
            :return: 该csv字符串的三轴时域曲线
            """
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            axis_time_domain_data = {
                "unit": csv_parsed_data["unit_after_scaling"]
            }
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                axis_time_domain_data[axis] = self.cal_time_data(
                    time_domain_curve_raw=csv_parsed_data[f"time_domain_curve_raw_{axis}"],
                    scale_val=csv_parsed_data["scale_data"][axis],
                    unit_after_scaling=csv_parsed_data["unit_after_scaling"],
                    samp_freq=csv_parsed_data["sampling_freq"]
                )
            return axis_time_domain_data

        def cal_freq_domain_data(self, time_domain_val: np.array, samp_freq: np.float64, code_window_type: str) -> dict:
            """
            输入曲线时域数据，根据采样频率，计算频域数据，包括频率、振幅、相位
            :param time_domain_val: 时域振动数值
            :return: 频域数据构成的字典对象
            """
            n = len(time_domain_val)

            # 去除直流分量
            time_domain_val_processed = time_domain_val - np.mean(time_domain_val)

            window = self.generate_window(signal_len=n, code_window_type=code_window_type)
            time_domain_val_processed = time_domain_val_processed * window

            # 去除加窗信号的直流分量
            time_domain_val_processed = time_domain_val_processed - np.mean(time_domain_val_processed)

            # T = 1 / samp_freq  # 采样间隔
            freqs = np.fft.fftfreq(n, 1 / samp_freq)
            # select_idxes = np.where(freqs >= 0)
            select_idxes = self.cal_select_idxes(n)
            freqs = freqs[select_idxes]

            fft_val = fft(time_domain_val_processed)
            # amplitude_response = (np.abs(fft_val)/n)[select_idxes]  # 归一化处理
            # phase_response = np.angle(fft_val)[range(freqs_len)]
            auto_power_linear = fft_val
            auto_power_linear /= n  # *2的作用是把负频率组分叠加到正频率，转换为单边数据，/n的作用是归一化
            auto_power_linear[1:] *= 2  # *2的作用是把负频率组分叠加到正频率，转换为单边数据，/n的作用是归一化
            auto_power_linear = np.abs(auto_power_linear)
            amplitude_response = auto_power_linear

            freq_res_len = int(n // 2.56)  # 由于混叠现象，高于采样频率/2.5的频带能量都会失真，再考虑到计算机的二进制存储，故通常取采样频率/2.56作为频带上限
            freqs = freqs[: freq_res_len]
            amplitude_response = amplitude_response[: freq_res_len]

            freq_domain_data = {
                "freqs": freqs,
                "amplitude_response": amplitude_response,
                # "phase_response": phase_response
            }
            return freq_domain_data

        def cal_time_data(self, time_domain_curve_raw: np.array, scale_val: np.float64, unit_after_scaling: str,
                          samp_freq: np.float64) -> dict:
            """
            输入csv获取的原始时域数据，得到时域曲线
            :param time_domain_curve_raw: 传感器获取的原始时域曲线数据
            :param scale_val: 尺度缩放参数
            :param unit_after_scaling: 传感器数据缩放后的单位，用于判断是否需要进一步处理。若单位为g，则需要继续转换到m/s^2
            :return: 包含时域曲线和频域曲线的字典对象
            """
            t_array = self.cal_t_array(n=len(time_domain_curve_raw), samp_freq=samp_freq)
            conversed_time_domain_curve_data = self.converse_unit(
                curve_data=time_domain_curve_raw, scale_val=scale_val, unit_after_scaling=unit_after_scaling
            )
            time_domain_data = {
                "time": keep_specified_decimal(t_array),
                "val": keep_specified_decimal(conversed_time_domain_curve_data),
                "normalized_val": keep_specified_decimal(self.cal_normalized_signal(conversed_time_domain_curve_data))
            }
            return time_domain_data

        def time_freq_data_process(self, time_domain_curve_raw: np.array, scale_val: np.float64,
                                   unit_after_scaling: str, samp_freq: np.float64, code_window_type: str) -> dict:
            """
            输入csv获取的原始时域数据，得到时域曲线和频域曲线
            :param time_domain_curve_raw: 传感器获取的原始时域曲线数据
            :param scale_val: 尺度缩放参数
            :param unit_after_scaling: 传感器数据缩放后的单位，用于判断是否需要进一步处理。若单位为g，则需要继续转换到m/s^2
            :param samp_freq: 采样频率
            :return: 包含时域曲线和频域曲线的字典对象
            """

            time_domain_data = self.cal_time_data(
                time_domain_curve_raw=time_domain_curve_raw,
                scale_val=scale_val, unit_after_scaling=unit_after_scaling,
                samp_freq=samp_freq
            )

            freq_domain_data = self.cal_freq_domain_data(
                time_domain_val=np.array(time_domain_data["normalized_val"]),
                samp_freq=samp_freq,
                code_window_type=code_window_type
            )

            # idx_peak, _ = find_peaks(freq_domain_data["amplitude_response"])
            idx_peak = self.get_top_ratio_peak_idx(freq_domain_data["amplitude_response"])
            n_spectrum = len(freq_domain_data["amplitude_response"])
            idx_nearest_peak = self.get_nearest_peak_idx(idx_peak=idx_peak, n_spectrum=n_spectrum)

            time_freq_data = {
                "time_domain": time_domain_data,
                "freq_domain": {
                    "freq": keep_specified_decimal(freq_domain_data["freqs"]),
                    "amplitude": keep_specified_decimal(freq_domain_data["amplitude_response"]),
                    "idx_peak": idx_peak,
                    "idx_nearest_peak": idx_nearest_peak
                }
            }
            return time_freq_data

        def deg_phase_trans(self, deg_phase_origin: np.array) -> np.array:
            deg_phase_trans_val = deg_phase_origin % 360
            deg_phase_trans_val[deg_phase_trans_val > 180] -= 360
            return deg_phase_trans_val

        def cross_phase_analyse(self, csv_data_str: str) -> dict:
            """
            输入原始传感器数据字符串，输出交叉相位分析结果字典对象
            :param csv_data_str: 原始传感器数据字符串
            :return: 交叉相位分析结果字典对象
            """
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            axis_time_domain_data = {}
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                axis_time_domain_data[axis] = self.cal_time_data(
                    time_domain_curve_raw=csv_parsed_data[f"time_domain_curve_raw_{axis}"],
                    scale_val=csv_parsed_data["scale_data"][axis],
                    unit_after_scaling=csv_parsed_data["unit_after_scaling"],
                    samp_freq=csv_parsed_data["sampling_freq"]
                )

            time_domain_val_x = axis_time_domain_data[CODE_AXIS_X]["normalized_val"]
            time_domain_val_y = axis_time_domain_data[CODE_AXIS_Y]["normalized_val"]
            time_domain_val_z = axis_time_domain_data[CODE_AXIS_Z]["normalized_val"]

            f_xy, p_xy = csd(time_domain_val_x, time_domain_val_y, fs=csv_parsed_data["sampling_freq"], nperseg=1024)
            f_yz, p_yz = csd(time_domain_val_y, time_domain_val_z, fs=csv_parsed_data["sampling_freq"], nperseg=1024)
            f_zx, p_zx = csd(time_domain_val_z, time_domain_val_x, fs=csv_parsed_data["sampling_freq"], nperseg=1024)

            n = len(time_domain_val_x)
            freq_res_len = int(n // 2.56)  # 由于混叠现象，高于采样频率/2.5的频带能量都会失真，再考虑到计算机的二进制存储，故通常取采样频率/2.56作为频带上限

            f_xy = f_xy[: freq_res_len]
            f_yz = f_yz[: freq_res_len]
            f_zx = f_zx[: freq_res_len]
            magnitude_xy = np.abs(p_xy)[: freq_res_len]
            phase_xy = self.deg_phase_trans(np.angle(p_xy, deg=True))[: freq_res_len]
            magnitude_yz = np.abs(p_yz)[: freq_res_len]
            phase_yz = self.deg_phase_trans(np.angle(p_yz, deg=True))[: freq_res_len]
            magnitude_zx = np.abs(p_zx)[: freq_res_len]
            phase_zx = self.deg_phase_trans(np.angle(p_zx, deg=True))[: freq_res_len]

            """
            n = len(time_domain_val_x)
            samp_freq = csv_parsed_data["sampling_freq"]
            T = 1 / samp_freq  # 采样间隔
            freqs = np.fft.fftfreq(n, T)
            # select_idxes = np.where(freqs >= 0)
            select_idxes = self.cal_select_idxes(n)
            freqs = freqs[select_idxes]

            fft_val_x = fft(time_domain_val_x)
            fft_val_y = fft(time_domain_val_y)
            fft_val_z = fft(time_domain_val_z)

            complementary_amplitude_response = (np.abs((fft_val_x + fft_val_y + fft_val_z) / 3) / n)[
                select_idxes]  # 归一化处理
            phase_response_diff_xy = (np.angle(fft_val_y, deg=True) - np.angle(fft_val_x, deg=True))[select_idxes]
            phase_response_diff_yz = (np.angle(fft_val_z, deg=True) - np.angle(fft_val_y, deg=True))[select_idxes]
            phase_response_diff_zx = (np.angle(fft_val_x, deg=True) - np.angle(fft_val_z, deg=True))[select_idxes]
            """

            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                # 创建子图布局
                fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                                    subplot_titles=["互功率密度谱CSD(Cross Spectral Density)", "交叉相位"])
                fig.add_trace(go.Scatter(x=f_xy, y=magnitude_xy, mode="lines", name=f"CSD XY"), row=1, col=1)
                fig.add_trace(go.Scatter(x=f_yz, y=magnitude_yz, mode="lines", name=f"CSD YZ"), row=1, col=1)
                fig.add_trace(go.Scatter(x=f_zx, y=magnitude_zx, mode="lines", name=f"CSD ZX"), row=1, col=1)

                fig.add_trace(go.Scatter(x=f_xy, y=phase_xy, mode="lines", name=f"Phase XY"), row=2, col=1)
                fig.add_trace(go.Scatter(x=f_yz, y=phase_yz, mode="lines", name=f"Phase YZ"), row=2, col=1)
                fig.add_trace(go.Scatter(x=f_zx, y=phase_zx, mode="lines", name=f"Phase ZX"), row=2, col=1)

                # 设置布局
                fig.update_layout(title="交叉相位信息展示", showlegend=True)

                # 显示图形
                fig.show()

                fig.write_html("交叉相位信息展示.html")

            cross_phase_analysis = {
                "freqs": self.post_process_data(keep_specified_decimal(f_xy)),
                "complementary_amplitude_response_xy": self.post_process_data(keep_specified_decimal(magnitude_xy)),
                "complementary_amplitude_response_yz": self.post_process_data(keep_specified_decimal(magnitude_yz)),
                "complementary_amplitude_response_zx": self.post_process_data(keep_specified_decimal(magnitude_zx)),
                "phase_response_diff_xy": self.post_process_data(keep_specified_decimal(phase_xy)),
                "phase_response_diff_yz": self.post_process_data(keep_specified_decimal(phase_yz)),
                "phase_response_diff_zx": self.post_process_data(keep_specified_decimal(phase_zx))
            }
            return cross_phase_analysis

        def cal_time_integral_data(self, signal: np.array, sampling_freq: np.float64) -> np.array:
            """
            计算时间积分数据
            :param signal: 待积分的时域数据，如加速度时域数据等
            :param sampling_freq: 采样频率
            :return:
            """
            n = len(signal)
            integral_data = np.zeros(n)
            for i in range(n - 1):
                integral_data[i + 1] = integral_data[i] + (signal[i] + signal[i + 1]) / 2
            integral_data /= sampling_freq
            return integral_data

        def cal_vibration_data(self, csv_data_str: str, start_time: np.float64 = SPEED_CURVE_START_TIME,
                               len_ratio: np.float64 = SPEED_CURVE_LEN_RATIO) -> dict:
            """
            输入原始数采的加速度时域数据字符串，输出速度时域信号数组
            :param csv_data_str: 原始数采的加速度时域数据字符串
            :return: 速度时域信号数组
            """
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            samp_freq = csv_parsed_data["sampling_freq"]
            vibration_data = {
                "unit": "mm/s",
                "sampling_freq": samp_freq
            }
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                cur_axis_time_domain_curve_raw = csv_parsed_data[f"time_domain_curve_raw_{axis}"]
                scale_val = csv_parsed_data["scale_data"][axis]
                # acc_signal = cur_axis_time_domain_curve_raw * scale_val
                acc_signal = cur_axis_time_domain_curve_raw * scale_val * GRAVITY_ACC  # 积分计算速度需要转换为标准国际单位
                acc_signal_mean = np.mean(acc_signal)

                speed_signal = self.cal_time_integral_data(signal=acc_signal - acc_signal_mean, sampling_freq=samp_freq)
                speed_signal *= 1000  # 换算单位到mm/s

                speed_signal_adjust = self.adjust_curve(
                    curve_raw=speed_signal, samp_freq=samp_freq, start_time=start_time, len_ratio=len_ratio
                )

                t_array = self.cal_t_array(n=len(speed_signal_adjust), samp_freq=samp_freq)

                vibration_data[axis] = {
                    "time_domain": {
                        "time": keep_specified_decimal(t_array),
                        "val": keep_specified_decimal(speed_signal_adjust),
                        "normalized_val": keep_specified_decimal(self.cal_normalized_signal(speed_signal_adjust))
                    }
                }
            return vibration_data

        def cal_signal_rms_val(self, csv_data_str: str, val_type: str) -> dict:
            """
            输入原始数采时域数据字符串，输出信号rms
            :param csv_data_str: 原始数采时域数据字符串
            :param val_type: 用于诊断的数据类型，如果是速度数据，需要从原始数据中积分出速度曲线后再做后续计算
            :return: 该信号的xyz三轴减去均值后的rms和数据的单位
            """
            if val_type == CODE_VAL_TYPE_VIBRATION:
                vibraton_data = self.cal_vibration_data(csv_data_str)
                rms_unit_data = {
                    "unit": vibraton_data["unit"],
                }
                rms = {}
                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    rms[axis] = self.cal_rms(signal=vibraton_data[axis]["time_domain"]["normalized_val"])
            else:
                csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
                rms_unit_data = {
                    "unit": csv_parsed_data["unit_after_scaling"]
                }
                rms = {}
                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    time_domain_curve_raw = csv_parsed_data[f"time_domain_curve_raw_{axis}"]
                    scale_val = csv_parsed_data["scale_data"][axis]
                    signal = time_domain_curve_raw * scale_val
                    signal_mean = np.mean(signal)
                    print(f"signal_{axis} mean: {signal_mean}")
                    rms[axis] = self.cal_rms(signal=signal - signal_mean)
            rms_unit_data["rms"] = rms
            return rms_unit_data

        def cal_signal_p2p_val(self, csv_data_str: str, val_type: str) -> dict:
            """
            输入原始数采时域数据字符串，输出信号峰峰值
            :param csv_data_str: 原始数采时域数据字符串
            :param val_type: 用于诊断的数据类型，如果是速度数据，需要从原始数据中积分出速度曲线后再做后续计算
            :return: 该信号的xyz三轴减去均值后的rms和数据的单位
            """
            if val_type == CODE_VAL_TYPE_VIBRATION:
                vibration_data = self.cal_vibration_data(csv_data_str)
                peak_to_peak_unit_data = {
                    "unit": vibration_data["unit"],
                }
                peak_to_peak = {}
                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    peak_to_peak[axis] = self.cal_p2p(signal=vibration_data[axis]["time_domain"]["normalized_val"])
            else:
                csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
                peak_to_peak_unit_data = {
                    "unit": csv_parsed_data["unit_after_scaling"]
                }
                peak_to_peak = {}
                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    time_domain_curve_raw = csv_parsed_data[f"time_domain_curve_raw_{axis}"]
                    scale_val = csv_parsed_data["scale_data"][axis]
                    signal = time_domain_curve_raw * scale_val
                    signal_mean = np.mean(signal)
                    print(f"signal_{axis} mean: {signal_mean}")
                    peak_to_peak[axis] = self.cal_p2p(signal=signal - signal_mean)
            peak_to_peak_unit_data["p2p"] = peak_to_peak
            return peak_to_peak_unit_data

        def cal_audio_signal_decibel_val(self, csv_data_str: str) -> dict:
            """
            输入原始数声音数据字符串，输出声音rms(单位Pa)，分贝数
            :param csv_data_str: 原始数采声音时域数据字符串
            :return: 声音数据rms，分贝数
            """
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            default_data_axis = CODE_AXIS_X

            audio_signal_decibel_data = {}
            audio_signal_time_domain_curve_raw = csv_parsed_data[f"time_domain_curve_raw_{default_data_axis}"]
            scale_val = csv_parsed_data["scale_data"][default_data_axis]
            signal = audio_signal_time_domain_curve_raw * scale_val

            """
            # 尝试输出音频文件
            # 将数据缩放到 16-bit 范围
            audio_data = np.int16(signal * 32767)
            from scipy.io.wavfile import write
            write("output.wav", np.int64(csv_parsed_data["sampling_freq"]), audio_data)
            """

            # signal_mean = np.mean(signal)
            # signal_rms = self.cal_rms(signal=signal - signal_mean)

            # 噪声数据有效值不能减去均值来处理，因为在有限的采样时间窗口里获得的噪声数据可能没有包含一个完整的变化周期，尤其是在抽油机上
            signal_rms = self.cal_rms(signal=signal)
            audio_signal_decibel_data["rms"] = signal_rms
            audio_signal_decibel_data["rmsUnit"] = csv_parsed_data["unit_after_scaling"]
            audio_signal_decibel_data["decibel"] = 20 * np.log10(signal_rms / P_REF)
            return audio_signal_decibel_data

        def cal_signal_feature_data(self, signal: np.array) -> dict:
            """
            输入信号数据，输出三轴时域数据的特征值字典
            :param signal:  已经过缩放和单位换算处理的三轴时域数据曲线
            :return:
            """
            try:
                signal_mean = np.mean(signal)
                signal_std = np.std(signal)
                # signal_adjusted = signal - signal_mean
                signal_adjusted = signal
                signal_max = np.max(signal_adjusted)
                signal_min = np.min(signal_adjusted)
                kurtosis_manual = np.sum((signal - signal_mean) ** 4) / (signal_std ** 4) / len(signal)  # 峭度，需要讨论应该用n还是n-1，目前看到的scipy库中计算其他指标用的是n
                # skewness_manual = np.sum((signal-signal_mean)**3)/(signal_std**3)/len(signal)  # 此处用公式计算的skew歪度，与scipy计算结果一致
                # skewness = skew(signal_adjusted)  # 偏度、歪度
                # rms = self.cal_rms(signal=signal_adjusted)  # 有效值，即均方根
                # mean_amplitude = np.mean(np.abs(signal_adjusted))  # 平均幅值
                skewness = skew(signal)  # 偏度、歪度
                rms = self.cal_rms(signal=signal)  # 有效值，即均方根
                mean_amplitude = np.mean(np.abs(signal))  # 平均幅值
                crest_factor = signal_max / rms  # 峰值因子，即ABB的默认指标
                shape_factor = rms / mean_amplitude  # 波形因子
                impulse_factor = signal_max / mean_amplitude  # 脉冲因子
                # root_square_amplitude = np.mean(np.sqrt(np.abs(signal_adjusted))) ** 2  # 方根幅值
                root_square_amplitude = np.mean(np.sqrt(np.abs(signal))) ** 2  # 方根幅值
                margin_factor = signal_max / root_square_amplitude  # 裕度因子
                """
                peaks, _ = find_peaks(signal_adjusted)
                valleys, _ = find_peaks(-signal_adjusted)
                peaks_and_valleys = np.sort(np.concatenate((peaks, valleys)))
                peak_diff = np.array([np.abs(signal_adjusted[peaks_and_valleys[idx]] - signal_adjusted[peaks_and_valleys[idx + 1]])
                             for idx in range(len(peaks_and_valleys) - 1)])
                peak_diff = peak_diff[peak_diff>=3]
                peak_val = np.mean(peak_diff)/2 # 峰值
                """
                peak_val = np.max(np.abs(signal_adjusted)) - mean_amplitude  # 峰值
                root_square_amplitude = np.mean(np.sqrt(np.abs(signal))) ** 2  # 方根幅值
                # peak_val = (signal_max-signal_min)/2
                peak_to_peak_val = signal_max - signal_min  # 峰峰值
                # energy = np.sum(signal_adjusted**2)  # 总能量

                signal_feature_data = {
                    "mean": handle_nan(data=signal_mean),
                    "max": handle_nan(data=signal_max),
                    "min": handle_nan(data=signal_min),
                    "rms": handle_nan(data=rms),  # 有效值，均方根
                    "kurtosis_val": handle_nan(data=kurtosis_manual),  # 峭度
                    "skewness_val": handle_nan(data=skewness),  # 偏度、歪度
                    "margin_factor": handle_nan(data=margin_factor),  # 裕度因子
                    "crest_factor": handle_nan(data=crest_factor),  # 峰值因子，即ABB的默认指标
                    "shape_factor": handle_nan(data=shape_factor),  # 波形因子
                    "impulse_factor": handle_nan(data=impulse_factor),  # 脉冲因子
                    "mean_amplitude": handle_nan(data=mean_amplitude),  # 平均幅值
                    "root_square_amplitude": handle_nan(data=root_square_amplitude),  # 方根幅值
                    "peak_val": handle_nan(data=peak_val),  # 峰值
                    "peak_to_peak_val": handle_nan(data=peak_to_peak_val)  # 峰峰值
                }
            except:
                signal_feature_data = None
            return signal_feature_data

        def cal_axis_signal_feature_data_by_curve(self, time_domain_curve_raw: dict, unit: str, scale_val: np.float64=1,
                                                  start_time: np.float64=0, len_ratio: np.float64=1) -> dict:
            """
            输入曲线数据字典对象，输出该数据XYZ轴的信号特征值
            :param time_domain_curve_raw:  原始的三轴时域数值数据
            :param unit:  时域数据单位
            :param scale_val:  缩放系数（为了与csv字符串的处理方式保持一致）
            :return:
            """
            axis_feature_data = {
                "unit": unit
            }
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                cur_axis_time_domain_curve_raw = time_domain_curve_raw[axis]
                signal = self.converse_unit(
                    curve_data=cur_axis_time_domain_curve_raw, scale_val=scale_val, unit_after_scaling=unit
                )

                signal_adjust = self.adjust_curve(
                    curve_raw=signal, samp_freq=time_domain_curve_raw["sampling_freq"], start_time=start_time, len_ratio=len_ratio
                )

                n = len(signal_adjust)
                get_time_len = n / time_domain_curve_raw["sampling_freq"]
                print(f"total time len: {get_time_len}")

                axis_feature_data[axis] = self.cal_signal_feature_data(signal=signal_adjust)
            return axis_feature_data

        def cal_axis_signal_feature_data(self, csv_data_str: str, val_type: str, start_time: np.float64=0, len_ratio: np.float64=1) -> dict:
            """
            输入原始数采时域数据字符串，输出各轴特征数据
            :param csv_data_str: 原始数采时域数据字符串
            :param val_type: 用于诊断的数据类型，如果是速度数据，需要从原始数据中积分出速度曲线后再做后续计算
            :return: 该信号的xyz三轴时域曲线各个特征数据和数据的单位
            """
            if val_type == CODE_VAL_TYPE_VIBRATION:
                if start_time == -1:
                    vibration_data = self.cal_vibration_data(csv_data_str=csv_data_str)
                else:
                    vibration_data = self.cal_vibration_data(csv_data_str=csv_data_str, start_time=start_time, len_ratio=len_ratio)
                time_domain_curve_raw = {
                    "sampling_freq": vibration_data["sampling_freq"]
                }
                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    time_domain_curve_raw[axis] = vibration_data[axis]["time_domain"]["val"]

                axis_feature_data = self.cal_axis_signal_feature_data_by_curve(
                    time_domain_curve_raw=time_domain_curve_raw,
                    unit=vibration_data["unit"],
                )
            else:
                csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
                unit_after_scaling = csv_parsed_data["unit_after_scaling"]
                axis_feature_data = {
                    "unit": unit_after_scaling
                }
                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    time_domain_curve_raw = csv_parsed_data[f"time_domain_curve_raw_{axis}"]
                    scale_val = csv_parsed_data["scale_data"][axis]
                    signal = self.converse_unit(
                        curve_data=time_domain_curve_raw, scale_val=scale_val, unit_after_scaling=unit_after_scaling
                    )

                    signal_adjust = self.adjust_curve(
                        curve_raw=signal,
                        samp_freq=csv_parsed_data["sampling_freq"],
                        start_time=start_time,
                        len_ratio=len_ratio
                    )

                    n = len(signal_adjust)
                    get_time_len = n / csv_parsed_data["sampling_freq"]
                    print(f"total time len: {get_time_len}")

                    axis_feature_data[axis] = self.cal_signal_feature_data(signal=signal_adjust)
            return axis_feature_data

        def cal_rms(self, signal: np.array) -> np.float64:
            """
            输入信号，计算rms
            :param signal: 信号数据
            :return: rms数值
            """
            rms_val = np.sqrt(np.mean(signal ** 2))
            return rms_val

        def cal_p2p(self, signal: np.array) -> np.float64:
            """
            输入信号，计算峰峰值
            :param signal: 信号数据
            :return: 峰峰值数值
            """
            signal_mean = np.mean(signal)
            signal_adjusted = signal - signal_mean
            signal_max = np.max(signal_adjusted)
            signal_min = np.min(signal_adjusted)
            peak_to_peak_val = signal_max-signal_min
            return peak_to_peak_val

        def cal_normalized_freq(self, input_freq: np.float64, samp_freq: np.float64) -> np.float64:
            """
            计算归一化频率
            :param input_freq: 输入真实频率
            :param samp_freq: 采样频率
            :return: 归一化频率
            """
            return 2 * input_freq / samp_freq

        def cal_normalized_signal(self, signal: np.array) -> np.float64:
            # sign_rms_val = np.sqrt(np.mean(signal**2))*np.sign(np.mean(signal))
            # sign_rms_val = cal_rms(signal=signal)*np.sign(np.mean(signal))
            # return signal-sign_rms_val
            sign_mean_val = np.mean(signal)
            return signal - sign_mean_val

        def filter_2(self, samp_freq: np.float64, freq: np.array, amplitude: np.array, cut_off_freq_low: np.float64,
                     cut_off_freq_high: np.float64) -> dict:
            """
            输入频谱(频率和振幅)，低通截止频率、高通截止频率，输出滤波后的时域图形
            :param samp_freq: 采样频率
            :param freq:
            :param amplitude:
            :param cut_off_freq_low:
            :param cut_off_freq_high:
            :return:
            """
            filtered_amplitude = amplitude
            filtered_amplitude[freq < cut_off_freq_low or freq > cut_off_freq_high] = 0
            return None

        def signal_processor_filter(self, input_data: np.array, samp_freq: np.float64, cut_off_freq_low: np.float64,
                                    cut_off_freq_high: np.float64, filter_func_code: np.int64, filter_order: int = 5) -> np.array:
            """
            执行滤波，返回滤波后的信号时域数组
            :param samp_freq:  采样频率
            :param filter_order:  滤波器阶数
            :param cut_off_freq_low:  低通截止频率
            :param cut_off_freq_high:  高通截止频率
            :param filter_func_code:  滤波功能代码
            :return:  滤波后的数据
            """
            if cut_off_freq_low < 0:
                raise Exception("低通截止不能小于0")
            if cut_off_freq_high > samp_freq * 0.5:
                raise Exception("高通截止不能大于带宽")

            wn_low = self.cal_normalized_freq(input_freq=cut_off_freq_low, samp_freq=samp_freq)  # 归一化截止频率
            wn_high = self.cal_normalized_freq(input_freq=cut_off_freq_high, samp_freq=samp_freq)  # 归一化截止频率

            # 配置滤波器 8 表示滤波器的阶数, filter_molecular_coef_vec为滤波器分子系数向量，filter_denominator_coef_vec为滤波器分母系数向量
            if filter_func_code == CODE_FILTER_BOND_PASS or filter_func_code == CODE_FILTER_BOND_STOP:
                if wn_low > 0:
                    if filter_func_code == CODE_FILTER_BOND_PASS:
                        wn_high = np.max([wn_high, FILTER_WN_CUT_LIMIT])  # 修正截止频率避免出现异常发散的滤波结果
                    filter_molecular_coef_vec, filter_denominator_coef_vec = ss.butter(
                        N=filter_order, Wn=[wn_low, wn_high], btype=DIC_FILTER_FUNC[filter_func_code]
                    )
                else:
                    if filter_func_code == CODE_FILTER_BOND_PASS:
                        wn = wn_high
                        wn = np.max([wn, FILTER_WN_CUT_LIMIT])
                        filter_molecular_coef_vec, filter_denominator_coef_vec = ss.butter(
                            N=filter_order, Wn=wn, btype="lowpass"
                        )
                    else:
                        filter_molecular_coef_vec, filter_denominator_coef_vec = ss.butter(
                            N=filter_order, Wn=wn_high, btype="highpass"
                        )
            else:
                if filter_func_code == CODE_FILTER_HIGH_PASS:
                    wn = wn_high
                else:
                    wn = wn_low
                    wn = np.max([wn, FILTER_WN_CUT_LIMIT])

                filter_molecular_coef_vec, filter_denominator_coef_vec = ss.butter(N=filter_order, Wn=wn, btype=DIC_FILTER_FUNC[filter_func_code])

            filtered_data = ss.filtfilt(b=filter_molecular_coef_vec, a=filter_denominator_coef_vec, x=input_data)
            return keep_specified_decimal(filtered_data)

        def get_top_ratio_peak_idx(self, data: np.array, ratio: np.float64 = 0.1, top_peak_len_min_limit: np.int64 = TOP_PEAK_LEN_DEFAULT) -> np.array:
            rms = self.cal_rms(signal=data)  # 有效值，即均方根
            # min_height = 0.2*peak_to_peak  # 波峰的最小高度
            # min_prominence = 0.2*peak_to_peak  # 波峰的最小显著性
            min_height = 0.1 * rms
            min_prominence = 0.1 * rms
            idx_peak, _ = find_peaks(data, height=min_height, prominence=min_prominence)
            # idx_peak, _ = find_peaks(data)
            peak_data_len = len(idx_peak)
            top_len = np.max([np.int64(np.ceil(peak_data_len) * ratio), np.min([top_peak_len_min_limit, peak_data_len])])
            peak_val = data[idx_peak]
            return idx_peak[np.argsort(-peak_val)[:top_len]]

        def get_nearest_peak_idx(self, idx_peak: np.array, n_spectrum: np.int64) -> np.array:
            """
            根据峰值谱线索引表和谱线数量，计算得到每个谱线最临近的峰值谱线的索引
            :param idx_peak: 峰值谱线索引表
            :param n_spectrum: 谱线数量
            :return: 记录每个谱线最临近的峰值谱线的索引表
            """
            if len(idx_peak) == 0:
                return np.array(range(n_spectrum))
            idx_peak = list(idx_peak)
            idx_peak.sort()
            idx_nearest_peak = np.zeros(n_spectrum)
            idx_nearest_peak[:idx_peak[0]] = idx_peak[0]
            idx_nearest_peak[idx_peak[-1]:] = idx_peak[-1]
            for i in range(len(idx_peak) - 1):
                cur_idx_peak = idx_peak[i]
                next_idx_peak = idx_peak[i + 1]
                mid = np.int64(np.round((cur_idx_peak + next_idx_peak) / 2))
                idx_nearest_peak[cur_idx_peak:mid] = cur_idx_peak
                idx_nearest_peak[mid:next_idx_peak] = next_idx_peak
            return idx_nearest_peak

        def generate_window(self, signal_len: np.int64, code_window_type: str = CODE_WINDOW_TYPE_DEFAULT) -> np.array:
            if code_window_type == CODE_WINDOW_TYPE_HAMMING:
                window = np.hamming(signal_len)
            elif code_window_type == CODE_WINDOW_TYPE_HANNING:
                window = np.hanning(signal_len)
            else:
                window = np.hanning(signal_len)  # 为避免异常，默认采用hanning窗
            return window

        def cal_signal_envelope_demodulation(self, signal: np.array, samp_freq: np.float64,
                                             cut_off_freq_low: np.float64, cut_off_freq_high: np.float64,
                                             code_window_type: str=CODE_WINDOW_TYPE_DEFAULT) -> dict:
            """
            输入原始信号数据，输出包络解调结果
            :param signal:  原始信号数据
            :param samp_freq:  采样频率
            :param cut_off_freq_low:  低通截止频率
            :param cut_off_freq_high:  高通截止频率
            :return:
            """
            n = len(signal)

            # hamming_window = np.hamming(n)  # 加hamming窗
            # signal_windowed = signal * hamming_window
            # hanning_window = np.hanning(n)  # 加hanning窗
            # signal_windowed = signal * hanning_window

            # 去除直流分量
            signal_processed = signal-np.mean(signal)

            # 加窗
            window = self.generate_window(signal_len=n, code_window_type=code_window_type)
            signal_processed = signal_processed * window

            # 去除加窗信号的直流分量
            signal_processed = signal_processed - np.mean(signal_processed)

            cut_off_freq_high = np.float64(np.min([cut_off_freq_high, samp_freq/2-1]))
            cut_off_freq_low = np.float64(np.max([cut_off_freq_low, 1]))

            filtered_signal = self.signal_processor_filter(
                input_data=signal_processed,
                # input_data=signal_windowed,
                samp_freq=samp_freq,
                cut_off_freq_low=cut_off_freq_low,
                cut_off_freq_high=cut_off_freq_high,
                filter_func_code=CODE_FILTER_BOND_PASS
            )
            filtered_signal = self.cal_normalized_signal(signal=filtered_signal)

            time_val = np.arange(0, n / samp_freq, 1 / samp_freq)

            """
            # 使用希尔伯特变换
            hilbert_transformed = hilbert(filtered_signal)

            # 计算包络
            hilbert_envelope_signal = np.abs(hilbert_transformed)
            hilbert_transformed_real = hilbert_transformed.real
            hilbert_transformed_imag = hilbert_transformed.imag
            """

            # 寻找信号的峰值
            peaks, _ = find_peaks(filtered_signal)
            peak_time = time_val[peaks]
            peak_val = filtered_signal[peaks]
            try:
                cubic_spline = CubicSpline(peak_time, peak_val, bc_type="not-a-knot")
                peak_envelope_signal = cubic_spline(time_val)
            except:
                peak_envelope_signal = filtered_signal

            # 测试观察滤波包络后的时域信号
            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                fig = go.FigureWidget()
                fig.add_trace(go.Scatter(x=time_val, y=signal, mode="lines", name="origin signal"))
                fig.add_trace(go.Scatter(x=time_val, y=filtered_signal, mode="lines", name="filtered signal"))
                # fig.add_trace(go.Scatter(x=time_val, y=hilbert_envelope_signal, mode="lines", name="hilbert envelope signal"))
                # fig.add_trace(go.Scatter(x=time_val, y=hilbert_transformed_real, mode="lines", name="hilbert transformed real"))
                # fig.add_trace(go.Scatter(x=time_val, y=hilbert_transformed_imag, mode="lines", name="hilbert transformed imag"))
                fig.add_trace(go.Scatter(x=time_val, y=peak_envelope_signal, mode="lines", name="peak envelope signal"))
                fig.show()

            T = 1 / samp_freq  # 采样间隔
            freqs = np.fft.fftfreq(n, T)

            """
            envelope_spectrum = np.abs(np.zeros_like(spectrum))
            abs_freqs = np.abs(freqs)
            select_idxes = np.where((abs_freqs>=cut_off_freq_low) | (abs_freqs<=cut_off_freq_high))
            envelope_spectrum[select_idxes] = spectrum[select_idxes]
            envelope_signal = np.abs(ifft(envelope_spectrum))
            """

            select_idxes = self.cal_select_idxes(n)
            # select_idxes = np.where(freqs>=0)

            freq_res_len = int(n // 2.56)  # 由于混叠现象，高于采样频率/2.5的频带能量都会失真，再考虑到计算机的二进制存储，故通常取采样频率/2.56作为频带上限

            freqs = freqs[select_idxes]
            freqs = freqs[: freq_res_len]

            # amplitude_response = (np.abs(fft(signal)) / n)[select_idxes]  # 归一化处理
            fft_val = np.fft.fft(signal_processed)[select_idxes]
            auto_power_linear = fft_val
            auto_power_linear /= n  # *2的作用是把负频率组分叠加到正频率，转换为单边数据，/n的作用是归一化
            auto_power_linear[1:] *= 2  # *2的作用是把负频率组分叠加到正频率，转换为单边数据，/n的作用是归一化
            auto_power_linear = np.abs(auto_power_linear)
            amplitude_response = auto_power_linear[: freq_res_len]
            # idx_peak_amplitude, _ = find_peaks(amplitude_response)
            idx_peak_amplitude = self.get_top_ratio_peak_idx(amplitude_response)
            n_spectrum = len(amplitude_response)
            idx_nearest_peak_amplitude = self.get_nearest_peak_idx(idx_peak=idx_peak_amplitude, n_spectrum=n_spectrum)

            # envelope_spectrum = (np.abs(fft(envelope_signal)))[select_idxes]
            # envelope_signal = cal_normalized_signal(envelope_signal)
            # hamming_window = np.hamming(n)  # 加hamming窗
            # peak_envelope_signal = peak_envelope_signal * hamming_window
            envelope_signal = self.cal_normalized_signal(peak_envelope_signal)
            # envelope_signal = peak_envelope_signal
            envelope_fft_val = np.fft.fft(envelope_signal)[select_idxes]
            auto_power_linear = envelope_fft_val
            auto_power_linear /= n  # *2的作用是把负频率组分叠加到正频率，转换为单边数据，/n的作用是归一化
            auto_power_linear[1:] *= 2  # *2的作用是把负频率组分叠加到正频率，转换为单边数据，/n的作用是归一化
            auto_power_linear = np.abs(auto_power_linear)
            envelope_spectrum = auto_power_linear[: freq_res_len]

            # 现在观察到，包络频谱分析中有能量泄漏到第一条谱线上，需要调研原因，此处先直接置0粗暴处理
            envelope_spectrum[0] = 0
            envelope_spectrum[1] = 0

            # idx_peak_envelope, _ = find_peaks(envelope_spectrum)
            idx_peak_envelope = self.get_top_ratio_peak_idx(envelope_spectrum)
            n_spectrum = len(envelope_spectrum)
            idx_nearest_peak_envelope = self.get_nearest_peak_idx(idx_peak=idx_peak_envelope, n_spectrum=n_spectrum)

            """
            print("show amplitude_response peaks value")
            print(amplitude_response[idx_peak_amplitude])
            print("show envelope_spectrum peaks value")
            print(envelope_spectrum[idx_peak_envelope])
            """

            # 测试观察滤波包络后的频域信号
            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                fig = go.FigureWidget()
                fig.add_trace(go.Scatter(x=freqs, y=envelope_spectrum, mode="lines", name="envelope demodulation"))
                fig.add_trace(go.Scatter(
                    x=freqs[idx_peak_envelope],
                    y=envelope_spectrum[idx_peak_envelope],
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name="包络峰值谱线"
                ))
                fig.show()

            envelope_demodulation = {
                "freq_domain": {
                    "freqs": keep_specified_decimal(freqs),
                    "amplitude_response": keep_specified_decimal(amplitude_response),
                    "idx_peak": idx_peak_amplitude,
                    "idx_nearest_peak": idx_nearest_peak_amplitude
                },
                "envelope_demodulation": {
                    # "time": cur_axis_time_domain_data["time"],
                    # "val": envelope_signal
                    "freqs": keep_specified_decimal(freqs),
                    "envelope_spectrum": keep_specified_decimal(envelope_spectrum),
                    "idx_peak": idx_peak_envelope,
                    "idx_nearest_peak": idx_nearest_peak_envelope
                },
            }

            """
            # 测试部分
            fig = go.FigureWidget()
            fig.add_trace(go.Scatter(x=freqs, y=amplitude_response, mode="lines", name="amplitude response"))
            fig.add_trace(go.Scatter(x=freqs, y=envelope_spectrum, mode="lines", name="envelope spectrum"))
            fig.show()
            """

            return envelope_demodulation

        def envelope_demodulate(self, csv_data_str: str, axis: str, cut_off_freq_low: np.float64,
                                cut_off_freq_high: np.float64, code_window_type: str=CODE_WINDOW_TYPE_DEFAULT) -> dict:
            """
            输入原始传感器数据字符串，低通截止频率，高通截止频率，输出包络解调计算结果
            :param csv_data_str: 原始传感器数据字符串
            :return: 包络解调结果字典对象
            """
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            cur_axis_time_domain_data = self.cal_time_data(
                time_domain_curve_raw=csv_parsed_data[f"time_domain_curve_raw_{axis}"],
                scale_val=csv_parsed_data["scale_data"][axis],
                unit_after_scaling=csv_parsed_data["unit_after_scaling"],
                samp_freq=csv_parsed_data["sampling_freq"]
            )
            # signal = cur_axis_time_domain_data["val"]
            signal = cur_axis_time_domain_data["normalized_val"]

            envelope_demodulation = self.cal_signal_envelope_demodulation(
                signal=signal,
                samp_freq=csv_parsed_data["sampling_freq"],
                cut_off_freq_low=cut_off_freq_low,
                cut_off_freq_high=cut_off_freq_high,
                code_window_type=code_window_type
            )
            envelope_demodulation["unit"] = csv_parsed_data["unit_after_scaling"]
            return envelope_demodulation

        def cal_select_idxes(self, n: np.int64) -> np.array:
            unilateral_len = (n + 1) // 2
            select_idxes = range(unilateral_len)
            return select_idxes

        def cal_signal_freq_spectrum_data(self, signal: np.array, samp_freq: np.float64, code_window_type: str=None) -> dict:
            """
            输入信号时域采样数组和采样频谱，可选窗函数类型，输出fft计算结果、线性自谱、平方自谱、PSD、ESD
            :param signal: 信号时域采样数组
            :param samp_freq: 采样频率
            :param code_window_type: 窗函数类型代码
            :return:
            """
            n = len(signal)
            if code_window_type is None:
                signal_processed = signal
            else:
                window = self.generate_window(signal_len=n, code_window_type=code_window_type)
                signal_processed = signal * window

            select_idxes = self.cal_select_idxes(n)
            freqs = np.fft.fftfreq(n, 1 / samp_freq)
            freqs = freqs[select_idxes]
            fft_val = np.fft.fft(signal_processed)[select_idxes]
            auto_power_linear = fft_val
            auto_power_linear /= n  # *2的作用是把负频率组分叠加到正频率，转换为单边数据，/n的作用是归一化
            auto_power_linear[1:] *= 2  # *2的作用是把负频率组分叠加到正频率，转换为单边数据，/n的作用是归一化
            auto_power_linear = np.abs(auto_power_linear)
            auto_power_power = auto_power_linear ** 2
            power_spectral_density = auto_power_power / (samp_freq / n)  # 用自功率谱除以频率分辨率（采样频率/采样点数），将不同频率分辨率下的分析结果以相同的标准进行比较
            energy_spectral_density = power_spectral_density * (n / samp_freq)

            signal_freq_spectrum_data = {
                "freqs": freqs,
                "fft_val": fft_val,
                "auto_power_linear": auto_power_linear,
                "auto_power_power": auto_power_power,
                "power_spectral_density": power_spectral_density,
                "energy_spectral_density": energy_spectral_density
            }
            return signal_freq_spectrum_data

        def cal_signal_freq_spectrum_envelope_demodulation(
                self, signal: np.array, samp_freq: np.float64, cut_off_freq_low: np.float64, cut_off_freq_high: np.float64,
                code_envelope_cal_mode: str=CODE_ENVELOPE_CAL_MODE_PEAK_CONECT, code_window_type: str=None) -> dict:
            """
            输入原始信号数据，输出包络解调结果
            :param signal: 原始信号数据
            :param samp_freq: 采样频率
            :param cut_off_freq_low: 低通截止频率
            :param cut_off_freq_high: 高通截止频率
            :param code_window_type: 窗函数类型代码
            :return:
            """

            n = len(signal)
            # 去除直流分量
            # signal_processed = signal-np.mean(signal)

            if code_window_type is None:
                signal_processed = signal
            else:
                # 加窗
                window = self.generate_window(signal_len=n, code_window_type=code_window_type)
                signal_processed = signal * window

            # 去除加窗信号的直流分量
            signal_processed = self.cal_normalized_signal(signal=signal_processed)

            cut_off_freq_high = np.float64(np.min([cut_off_freq_high, samp_freq/2-1]))
            cut_off_freq_low = np.float64(np.max([cut_off_freq_low, 1]))

            filtered_signal = self.signal_processor_filter(
                input_data=signal_processed,
                samp_freq=samp_freq,
                cut_off_freq_low=cut_off_freq_low,
                cut_off_freq_high=cut_off_freq_high,
                filter_func_code=CODE_FILTER_BOND_PASS
            )

            # 去除信号处理后的直流分量
            filtered_signal = self.cal_normalized_signal(signal=filtered_signal)
            time_val = np.arange(0, n / samp_freq, 1 / samp_freq)

            # 使用希尔伯特变换
            hilbert_transformed = hilbert(filtered_signal)

            # 希尔伯特变化计算包络
            hilbert_envelope_signal = np.abs(hilbert_transformed)
            hilbert_transformed_real = hilbert_transformed.real
            hilbert_transformed_imag = hilbert_transformed.imag

            # 寻找信号的峰值
            peaks, _ = find_peaks(filtered_signal)
            peak_time = time_val[peaks]
            peak_val = filtered_signal[peaks]
            try:
                cubic_spline = CubicSpline(peak_time, peak_val, bc_type="not-a-knot")
                peak_connect_envelope_signal = cubic_spline(time_val)
            except:
                peak_connect_envelope_signal = filtered_signal

            if code_envelope_cal_mode == CODE_ENVELOPE_CAL_MODE_HILBERT:
                peak_envelope_signal = hilbert_envelope_signal
            elif code_envelope_cal_mode == CODE_ENVELOPE_CAL_MODE_PEAK_CONECT:
                peak_envelope_signal = peak_connect_envelope_signal
            else:
                peak_envelope_signal = filtered_signal

            # 测试观察滤波包络后的时域信号
            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                fig = go.FigureWidget()
                fig.add_trace(go.Scatter(x=time_val, y=signal, mode="lines", name="origin signal"))
                fig.add_trace(go.Scatter(x=time_val, y=filtered_signal, mode="lines", name="filtered signal"))
                fig.add_trace(go.Scatter(x=time_val, y=hilbert_envelope_signal, mode="lines", name="hilbert envelope signal"))
                fig.add_trace(go.Scatter(x=time_val, y=hilbert_transformed_real, mode="lines", name="hilbert transformed real"))
                fig.add_trace(go.Scatter(x=time_val, y=hilbert_transformed_imag, mode="lines", name="hilbert transformed imag"))
                fig.add_trace(go.Scatter(x=time_val, y=peak_connect_envelope_signal, mode="lines", name="peak connect envelope signal"))
                fig.add_trace(go.Scatter(x=time_val, y=peak_envelope_signal, mode="lines", name="peak envelope signal"))
                fig.show()

            freq_res_len = int(n // 2.56)  # 由于混叠现象，高于采样频率/2.5的频带能量都会失真，再考虑到计算机的二进制存储，故通常取采样频率/2.56作为频带上限

            # 计算信号本身和峰值包络曲线的PSD等数据
            signal_freq_spectrum_data = self.cal_signal_freq_spectrum_data(signal=signal_processed, samp_freq=samp_freq)
            peak_envelope_signal = self.cal_normalized_signal(peak_envelope_signal)  # 去除直流分量
            peak_envelope_signal_freq_spectrum_data = self.cal_signal_freq_spectrum_data(signal=peak_envelope_signal, samp_freq=samp_freq)

            freqs = signal_freq_spectrum_data["freqs"][:freq_res_len]
            signal_power_spectral_density = signal_freq_spectrum_data["power_spectral_density"][:freq_res_len]
            idx_peak_amplitude = self.get_top_ratio_peak_idx(signal_power_spectral_density)

            peak_envelope_signal_power_spectral_density = peak_envelope_signal_freq_spectrum_data["power_spectral_density"][:freq_res_len]
            idx_peak_envelope = self.get_top_ratio_peak_idx(peak_envelope_signal_power_spectral_density)

            # 测试观察滤波包络后的频域信号
            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                fig = go.FigureWidget()
                fig.add_trace(go.Scatter(x=freqs, y=peak_envelope_signal_power_spectral_density, mode="lines", name="envelope demodulation PSD"))
                fig.add_trace(go.Scatter(
                    x=freqs[idx_peak_envelope],
                    y=peak_envelope_signal_power_spectral_density[idx_peak_envelope],
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name="包络信号PSD峰值"
                ))
                fig.show()

            data_envelope_demodulation = {
                "freq_domain": {
                    "freqs": keep_specified_decimal(freqs),
                    "amplitude_response_psd": keep_specified_decimal(signal_power_spectral_density),
                    # "idx_peak": idx_peak_amplitude,
                    # "idx_nearest_peak": idx_nearest_peak_amplitude
                },
                "envelope_demodulation": {
                    "freqs": keep_specified_decimal(freqs),
                    "envelope_spectrum_psd": keep_specified_decimal(peak_envelope_signal_power_spectral_density),
                    # "idx_peak": idx_peak_envelope,
                    # "idx_nearest_peak": idx_nearest_peak_envelope
                },
            }
            return data_envelope_demodulation

        @__check
        def cal_data_envelope_demodulation(self, csv_data_str: str, axis: str, cut_off_freq_low: np.float64, cut_off_freq_high: np.float64,
                                           code_envelope_cal_mode:str=CODE_ENVELOPE_CAL_MODE_PEAK_CONECT, code_window_type: str=None):
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            samp_freq = csv_parsed_data["sampling_freq"]
            code_axis = get_code_axis(axis=axis)
            cur_axis_time_domain_data = self.cal_time_data(
                time_domain_curve_raw=csv_parsed_data[f"time_domain_curve_raw_{code_axis}"],
                scale_val=csv_parsed_data["scale_data"][code_axis],
                unit_after_scaling=csv_parsed_data["unit_after_scaling"],
                samp_freq=samp_freq
            )
            signal = cur_axis_time_domain_data["val"]
            cur_axis_envelope_demodulation = self.cal_signal_freq_spectrum_envelope_demodulation(
                signal=signal,
                samp_freq=samp_freq,
                cut_off_freq_low=cut_off_freq_low,
                cut_off_freq_high=cut_off_freq_high,
                code_envelope_cal_mode=code_envelope_cal_mode,
                code_window_type=code_window_type
            )
            return cur_axis_envelope_demodulation

        def show_freq_analysis(self, csv_data_str1: str, csv_data_str2: str) -> None:
            csv_parsed_data1 = self.parse_data(csv_data_str=csv_data_str1)
            csv_parsed_data2 = self.parse_data(csv_data_str=csv_data_str2)

            axis_time_domain_data1 = {}
            axis_time_domain_data2 = {}
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                axis_time_domain_data1[axis] = self.cal_time_data(
                    time_domain_curve_raw=csv_parsed_data1[f"time_domain_curve_raw_{axis}"],
                    scale_val=csv_parsed_data1["scale_data"][axis],
                    unit_after_scaling=csv_parsed_data1["unit_after_scaling"],
                    samp_freq=csv_parsed_data1["sampling_freq"]
                )
                axis_time_domain_data2[axis] = self.cal_time_data(
                    time_domain_curve_raw=csv_parsed_data2[f"time_domain_curve_raw_{axis}"],
                    scale_val=csv_parsed_data2["scale_data"][axis],
                    unit_after_scaling=csv_parsed_data2["unit_after_scaling"],
                    samp_freq=csv_parsed_data2["sampling_freq"]
                )

            samp_freq1 = csv_parsed_data1["sampling_freq"]
            samp_freq2 = csv_parsed_data2["sampling_freq"]
            axis_frequency_domain_data1 = {}
            axis_frequency_domain_data2 = {}
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                n = len(axis_time_domain_data1[axis]["val"])
                n = np.int64(np.round(n / samp_freq2 * samp_freq1))
                axis_time_domain_data1[axis]["time"] = axis_time_domain_data1[axis]["time"][:n]
                axis_time_domain_data1[axis]["val"] = axis_time_domain_data1[axis]["val"][:n]
                axis_time_domain_data1[axis]["normalized_val"] = axis_time_domain_data1[axis]["normalized_val"][:n]
                # signal = np.array(axis_time_domain_data1[axis]["normalized_val"])
                signal = np.array(axis_time_domain_data1[axis]["val"])
                axis_frequency_domain_data1[axis] = self.cal_signal_freq_spectrum_data(signal=signal, samp_freq=samp_freq1)

                n = len(axis_time_domain_data2[axis]["val"])
                # signal = np.array(axis_time_domain_data2[axis]["normalized_val"])
                signal = np.array(axis_time_domain_data2[axis]["val"])
                axis_frequency_domain_data2[axis] = self.cal_signal_freq_spectrum_data(signal=signal, samp_freq=samp_freq2)

            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                # 创建子图布局
                fig = make_subplots(rows=2, cols=1, shared_xaxes=False, subplot_titles=["时域信息", "频域信息"])
                # 时域信息
                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    trace_time_domain = go.Scatter(
                        x=axis_time_domain_data1[axis]["time"],
                        # y=axis_time_domain_data[axis]["val"],
                        y=self.cal_normalized_signal(np.array(axis_time_domain_data1[axis]["val"])),  # 展示均值移动到0点的信号
                        mode="lines", name=f"低频加速度{axis}轴时域信号"
                    )
                    fig.add_trace(trace_time_domain, row=1, col=1)

                    trace_time_domain = go.Scatter(
                        x=axis_time_domain_data2[axis]["time"],
                        # y=axis_time_domain_data[axis]["val"],
                        y=self.cal_normalized_signal(np.array(axis_time_domain_data2[axis]["val"])),  # 展示均值移动到0点的信号
                        mode="lines", name=f"高频加速度{axis}轴时域信号"
                    )
                    fig.add_trace(trace_time_domain, row=1, col=1)

                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    trace_frequency_domain = go.Scatter(
                        x=axis_frequency_domain_data1[axis]["freqs"],
                        y=np.abs(axis_frequency_domain_data1[axis]["auto_power_linear"]),
                        mode="lines", name=f"低频加速度{axis}轴线性自谱"
                    )
                    fig.add_trace(trace_frequency_domain, row=2, col=1)

                    trace_frequency_domain = go.Scatter(
                        x=axis_frequency_domain_data2[axis]["freqs"],
                        y=np.abs(axis_frequency_domain_data2[axis]["auto_power_linear"]),
                        mode="lines", name=f"高频加速度{axis}轴线性自谱"
                    )
                    fig.add_trace(trace_frequency_domain, row=2, col=1)

                    trace_frequency_domain = go.Scatter(
                        x=axis_frequency_domain_data1[axis]["freqs"],
                        y=np.abs(axis_frequency_domain_data1[axis]["auto_power_power"]),
                        mode="lines", name=f"低频加速度{axis}轴平方自谱"
                    )
                    fig.add_trace(trace_frequency_domain, row=2, col=1)

                    trace_frequency_domain = go.Scatter(
                        x=axis_frequency_domain_data2[axis]["freqs"],
                        y=np.abs(axis_frequency_domain_data2[axis]["auto_power_power"]),
                        mode="lines", name=f"高频加速度{axis}轴平方自谱"
                    )
                    fig.add_trace(trace_frequency_domain, row=2, col=1)

                    trace_frequency_domain = go.Scatter(
                        x=axis_frequency_domain_data1[axis]["freqs"],
                        y=np.abs(axis_frequency_domain_data1[axis]["power_spectral_density"]),
                        mode="lines", name=f"低频加速度{axis}轴PSD"
                    )
                    fig.add_trace(trace_frequency_domain, row=2, col=1)

                    trace_frequency_domain = go.Scatter(
                        x=axis_frequency_domain_data2[axis]["freqs"],
                        y=np.abs(axis_frequency_domain_data2[axis]["power_spectral_density"]),
                        mode="lines", name=f"高频加速度{axis}轴PSD"
                    )
                    fig.add_trace(trace_frequency_domain, row=2, col=1)

                    trace_frequency_domain = go.Scatter(
                        x=axis_frequency_domain_data1[axis]["freqs"],
                        y=np.abs(axis_frequency_domain_data1[axis]["energy_spectral_density"]),
                        mode="lines", name=f"低频加速度{axis}轴ESD"
                    )
                    fig.add_trace(trace_frequency_domain, row=2, col=1)

                    trace_frequency_domain = go.Scatter(
                        x=axis_frequency_domain_data2[axis]["freqs"],
                        y=np.abs(axis_frequency_domain_data2[axis]["energy_spectral_density"]),
                        mode="lines", name=f"高频加速度{axis}轴ESD"
                    )
                    fig.add_trace(trace_frequency_domain, row=2, col=1)

                # 设置布局
                fig.update_layout(title="时域和频域信息展示", showlegend=True)

                # 显示图形
                fig.show()

        def diagnose_abnormal_peak_to_peak(self, signal: np.array) -> bool:
            """
            输入信号数据，诊断是否出现异常峰峰值
            :param signal: 诊断的目标信号数据数值
            :return: 表征是否出现异常峰峰值的布尔值，True为出现异常峰峰值
            """
            signal_mean = np.mean(signal)
            # signal_std = np.std(signal)
            signal_adjusted = signal - signal_mean

            signal_max = np.max(signal_adjusted)
            signal_min = np.min(signal_adjusted)
            # peak_to_peak = signal_max-signal_min

            rms = self.cal_rms(signal=signal_adjusted)  # 有效值，即均方根
            # min_height = 0.2*peak_to_peak  # 波峰的最小高度
            # min_prominence = 0.2*peak_to_peak  # 波峰的最小显著性
            min_height = 0.5 * rms
            min_prominence = 0.5 * rms

            # 找到波峰
            peaks, _ = find_peaks(signal_adjusted, height=min_height, prominence=min_prominence)

            # 找到波谷
            valleys, _ = find_peaks(-signal_adjusted, height=min_height, prominence=min_prominence)  # 波谷可以通过对信号取负找到

            # 计算波峰和相邻波谷的差值
            differences = []
            for peak in peaks:
                # 找到相邻的波谷
                if peak > 0 and peak < len(signal_adjusted) - 1:
                    # 查找上一个波谷
                    left_valley = valleys[valleys < peak].max() if np.any(valleys < peak) else None
                    # 查找下一个波谷
                    right_valley = valleys[valleys > peak].min() if np.any(valleys > peak) else None

                    # 计算差值并存储
                    if left_valley is not None:
                        diff_left = signal_adjusted[peak] - signal_adjusted[left_valley]
                        differences.append(diff_left)
                    if right_valley is not None:
                        diff_right = signal_adjusted[peak] - signal_adjusted[right_valley]
                        differences.append(diff_right)

            # 找到最大的差值
            max_difference = max(differences) if differences else None
            print("最大的波峰和相邻波谷的差值:", max_difference)

            # 可视化信号
            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                fig = go.FigureWidget()
                fig.add_trace(
                    go.Scatter(x=np.arange(len(signal_adjusted)), y=signal_adjusted, mode="lines", name="signal"))
                # 添加波峰
                fig.add_trace(
                    go.Scatter(
                        x=peaks,
                        y=signal_adjusted[peaks],
                        mode='markers',
                        marker=dict(color='red', size=10),
                        name='波峰'
                    )
                )
                # 添加波谷
                fig.add_trace(
                    go.Scatter(
                        x=valleys,
                        y=signal_adjusted[valleys],
                        mode='markers',
                        marker=dict(color='blue', size=10),
                        name='波谷'
                    )
                )

                # 更新布局
                fig.update_layout(
                    title='信号波峰和波谷',
                    xaxis_title='时间',
                    yaxis_title='信号强度',
                    showlegend=True
                )
                fig.show()

            mean_difference = np.mean(differences) if differences else None
            print("波峰和相邻波谷的差值的均值:", mean_difference)
            # rms = self.cal_rms(signal=signal_adjusted) # 有效值，即均方根
            # print(f"rms: {rms}")
            if max_difference > mean_difference * 2:
                if max_difference > 0.1:
                    return True
                else:
                    return False
            else:
                return False

        def diagnose_camera_move(self, csv_data_str: str) -> bool:
            """
            输入原始数采时域数据字符串(低频加速度或高频加速度)，诊断摄像机是否被人故意移动
            :param csv_data_str: 原始数采时域数据字符串
            :return: 表征是否被移动的布尔值，True代表诊断为摄像机已经被人移动
            """
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            unit_after_scaling = csv_parsed_data["unit_after_scaling"]
            detect_camera_move = False
            for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                time_domain_curve_raw = csv_parsed_data[f"time_domain_curve_raw_{axis}"]
                scale_val = csv_parsed_data["scale_data"][axis]
                signal = self.converse_unit(
                    curve_data=time_domain_curve_raw, scale_val=scale_val, unit_after_scaling=unit_after_scaling
                )
                signal_axis_abnormal_peak_to_peak = self.diagnose_abnormal_peak_to_peak(signal=signal)
                detect_camera_move = detect_camera_move or signal_axis_abnormal_peak_to_peak
            return detect_camera_move

        def idx_peak_merge(self, data: np.array, idx_peak: np.array) -> np.array:
            """
            输入信号值和峰值索引，把相邻的峰值索引合并到最大值，输出合并后的峰值索引
            :param data:
            :param idx_peak:
            :return:
            """
            merge_idx_peak = []
            temp_idx_peak = []
            check_range = 5  # 合并索引的前后邻域范围
            # check_val = np.mean(data[idx_peak[1:]])
            check_val = np.mean(data[idx_peak[1:]]) * 0.8
            for i in range(len(idx_peak)):
                cur_idx_peak = idx_peak[i]
                if data[cur_idx_peak] > check_val:
                    temp_idx_peak.append(cur_idx_peak)
                else:
                    break

            temp_idx_peak.sort()
            for i in range(len(temp_idx_peak)-1):
                cur_idx_peak = temp_idx_peak[i]
                if len(merge_idx_peak) == 0:
                    merge_idx_peak.append(cur_idx_peak)
                else:
                    if np.abs(merge_idx_peak[-1]-cur_idx_peak) < check_range:
                        if data[cur_idx_peak] > data[merge_idx_peak[-1]]:
                            merge_idx_peak[-1] = cur_idx_peak
                    else:
                        merge_idx_peak.append(cur_idx_peak)

            return np.array(merge_idx_peak)

        def diagnose_equipment(self, csv_data_str: str, equip_type: str, supp_data: dict=None) -> str:
            """
            输入原始数据和设备类型，输出诊断结果
            :param csv_data_str: 传感器原始数据
            :param equip_type: 诊断的设备类型
            :param supp_data: 诊断时需要额外传入的数据
            :return:
            """
            # 先根据unit_after_scaling判断是否是声音数据，再根据声音数据阈值诊断是否有异常振动噪声
            csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
            unit_after_scaling = csv_parsed_data["unit_after_scaling"]
            if unit_after_scaling == "Pa":
                time_domain_curve_raw = csv_parsed_data[f"time_domain_curve_raw_{CODE_AXIS_X}"]
                scale_val = csv_parsed_data["scale_data"][CODE_AXIS_X]
                signal = self.converse_unit(
                    curve_data=time_domain_curve_raw, scale_val=scale_val, unit_after_scaling=unit_after_scaling
                )
                signal_rms = self.cal_rms(signal=signal)
                audio_signal_decibel_val = 20 * np.log10(signal_rms / P_REF)
                if audio_signal_decibel_val < THRESHOLD_NOISE_STATUS_UP_NORMAL:
                    return CODE_CONDITION_NORMAL
                elif audio_signal_decibel_val < THRESHOLD_NOISE_STATUS_UP_ABNORMAL_LOW:
                    return CODE_CONDITION_NOISE_ABNORMAL_LOW
                elif audio_signal_decibel_val < THRESHOLD_NOISE_STATUS_UP_ABNORMAL_MID:
                    return CODE_CONDITION_NOISE_ABNORMAL_MID
                else:
                    return CODE_CONDITION_NOISE_ABNORMAL_HIGH

            if equip_type == CODE_EQUIP_TYPE_MOTOR:
                # csv_parsed_data = self.parse_data(csv_data_str=csv_data_str)
                # unit_after_scaling = csv_parsed_data["unit_after_scaling"]
                for axis in [CODE_AXIS_X, CODE_AXIS_Y, CODE_AXIS_Z]:
                    time_domain_curve_raw = csv_parsed_data[f"time_domain_curve_raw_{axis}"]
                    scale_val = csv_parsed_data["scale_data"][axis]
                    signal = self.converse_unit(
                        curve_data=time_domain_curve_raw, scale_val=scale_val, unit_after_scaling=unit_after_scaling
                    )

                    # signal_normalized = self.cal_normalized_signal(signal=signal)
                    signal_normalized = signal  # 抽油机等设备在采样区间内没有采集完整周期数据，因此不会呈现0附近的周期变化，故不用做均值修正

                    # 包络解调
                    cut_off_freq_low = 1500
                    cur_off_freq_high = 2500
                    code_window_type = CODE_WINDOW_TYPE_DEFAULT
                    envelope_demodulation = self.cal_signal_envelope_demodulation(
                        signal=signal_normalized,
                        samp_freq=csv_parsed_data["sampling_freq"],
                        cut_off_freq_low=cut_off_freq_low,
                        cut_off_freq_high=cur_off_freq_high,
                        code_window_type=code_window_type
                    )

                    """
                    envelope_demodulation = {
                        "freq_domain": {
                            "freqs": keep_specified_decimal(freqs),
                            "amplitude_response": keep_specified_decimal(amplitude_response),
                            "idx_peak": idx_peak_amplitude,
                            "idx_nearest_peak": idx_nearest_peak_amplitude
                        },
                        "envelope_demodulation": {
                            # "time": cur_axis_time_domain_data["time"],
                            # "val": envelope_signal
                            "freqs": keep_specified_decimal(freqs),
                            "envelope_spectrum": keep_specified_decimal(envelope_spectrum),
                            "idx_peak": idx_peak_envelope,
                            "idx_nearest_peak": idx_nearest_peak_envelope
                        },
                    }
                    """
                    freqs = envelope_demodulation["envelope_demodulation"]["freqs"]
                    envelope_spectrum = envelope_demodulation["envelope_demodulation"]["envelope_spectrum"]
                    idx_peak = envelope_demodulation["envelope_demodulation"]["idx_peak"]
                    # judge_val = np.mean(envelope_spectrum[idx_peak[1:]])*0.8
                    # temp = envelope_spectrum[idx_peak] > judge_val
                    # idx_peak = idx_peak[temp]
                    merge_idx_peak = self.idx_peak_merge(data=envelope_spectrum, idx_peak=idx_peak)

                    # 测试观察滤波包络后再经过筛选的频域信号
                    if ALLOW_SHOW_FIGURE:
                        import plotly.graph_objects as go
                        fig = go.FigureWidget()
                        fig.add_trace(
                            go.Scatter(x=freqs, y=envelope_spectrum, mode="lines", name="envelope demodulation"))
                        fig.add_trace(go.Scatter(
                            x=freqs[merge_idx_peak],
                            y=envelope_spectrum[merge_idx_peak],
                            mode="markers",
                            marker=dict(color="red", size=10),
                            name="包络峰值谱线"
                        ))
                        fig.show()

                    # idx_peak.sort()
                    if len(merge_idx_peak) > 5:
                        if merge_idx_peak[0] > 5:
                            idx_peak_base = merge_idx_peak[0]
                            idx_peak_check = merge_idx_peak[1:5]
                        else:
                            idx_peak_base = merge_idx_peak[1]
                            idx_peak_check = merge_idx_peak[2:6]

                        freq_multi = np.round(idx_peak_check / idx_peak_base)  # 检测倍频，如果有倍频，freq_multi应等于[2, 3, 4, 5]
                        check_cnt = 0
                        for i in range(len(freq_multi)):
                            if np.abs(freq_multi[i]-(i+2)) < 0.5:
                                check_cnt += 1
                        if check_cnt > len(freq_multi)/2:
                            return CODE_CONDITION_BEARING_WEAR
                return CODE_CONDITION_NORMAL  # 暂时简化处理，先调通接口，再细化功能
            elif equip_type == CODE_EQUIP_TYPE_BEARING:
                code = CODE_CONDITION_NORMAL
                if supp_data is None:
                    return code

                if "bearingStatus" in supp_data:
                    bearing_status = supp_data["bearingStatus"]
                    if bearing_status < THRESHOLD_BEARING_STATUS_UP_NORMAL:
                        code_by_bearing_status = CODE_CONDITION_NORMAL
                    elif bearing_status < THRESHOLD_BEARING_STATUS_UP_ABNORMAL_LOW:
                        code_by_bearing_status = CODE_CONDITION_BEARING_ABNORMAL_LOW
                    elif bearing_status < THRESHOLD_BEARING_STATUS_UP_ABNORMAL_MID:
                        code_by_bearing_status = CODE_CONDITION_BEARING_ABNORMAL_MID
                    else:
                        code_by_bearing_status = CODE_CONDITION_BEARING_ABNORMAL_HIGH

                if ROD_SPD_ACTUAL_INACCURATE:
                    """
                    if DICT_RISK_STATUS[code_by_bearing_status] > DICT_RISK_STATUS[CODE_CONDITION_NORMAL]:
                        code = code_by_bearing_status
                        return code
                    """
                    code = code_by_bearing_status
                    return code

                val_type = supp_data["valType"]
                if val_type == CODE_VAL_TYPE_VIBRATION:
                    judge_data = self.cal_signal_rms_val(csv_data_str=csv_data_str, val_type=val_type)
                    judge_val = np.max([judge_data["rms"][CODE_AXIS_X], judge_data["rms"][CODE_AXIS_Y], judge_data["rms"][CODE_AXIS_Z]])
                else:
                    judge_data = self.cal_signal_p2p_val(csv_data_str=csv_data_str, val_type=val_type)
                    judge_val = np.max([judge_data["p2p"][CODE_AXIS_X], judge_data["p2p"][CODE_AXIS_Y], judge_data["p2p"][CODE_AXIS_Z]])

                if val_type == CODE_VAL_TYPE_VIBRATION:
                    motor_rated_power = supp_data["motorRatedPower"]
                    if motor_rated_power < 15:
                        threshold_up_normal = 0.7
                        threshold_up_abnormal_low = 1.8
                        threshold_up_abnormal_mid = 4.5
                    elif motor_rated_power < 75:
                        threshold_up_normal = 1.1
                        threshold_up_abnormal_low = 2.8
                        threshold_up_abnormal_mid = 7.1
                    else:
                        threshold_up_normal = 1.8
                        threshold_up_abnormal_low = 4.5
                        threshold_up_abnormal_mid = 11
                else:
                    rot_spd_actual = supp_data["rotSpdActual"]
                    if rot_spd_actual < 500:
                        threshold_up_normal = 0.50
                        threshold_up_abnormal_low = 1.0
                        threshold_up_abnormal_mid = 2.0
                    elif rot_spd_actual < 1800:
                        threshold_up_normal = 0.75
                        threshold_up_abnormal_low = 2.0
                        threshold_up_abnormal_mid = 4.0
                    else:
                        threshold_up_normal = 1.0
                        threshold_up_abnormal_low = 3.0
                        threshold_up_abnormal_mid = 6.0
                if judge_val < threshold_up_normal:
                    code_by_judge_val = CODE_CONDITION_NORMAL
                elif judge_val < threshold_up_abnormal_low:
                    code_by_judge_val = CODE_CONDITION_BEARING_ABNORMAL_LOW
                elif judge_val < threshold_up_abnormal_mid:
                    code_by_judge_val = CODE_CONDITION_BEARING_ABNORMAL_MID
                else:
                    code_by_judge_val = CODE_CONDITION_BEARING_ABNORMAL_HIGH

                if DICT_RISK_STATUS[code_by_judge_val] < DICT_RISK_STATUS[code_by_bearing_status]:
                    code = code_by_bearing_status
                else:
                    code = code_by_judge_val
                return code
            else:
                return CODE_CONDITION_NORMAL

        def show_signal(self, x: np.array, y: np.array) -> None:
            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                fig = go.FigureWidget()
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode="lines"))
                fig.show()

        def show_data(self, data_list: list) -> None:
            if ALLOW_SHOW_FIGURE:
                import plotly.graph_objects as go
                fig = go.FigureWidget()
                for data in data_list:
                    fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode="lines", name=data["name"]))
                fig.show()

    return __Signal_Processor