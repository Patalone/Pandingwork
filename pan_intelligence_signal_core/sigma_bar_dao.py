import pymysql
from typing import Tuple, Optional, List
import json
import numpy as np
from typing import Dict

class SigmaBarDAO:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            autocommit=True
        )

    def close(self):
        self.conn.close()

    def _get_cursor(self):
        return self.conn.cursor()

    def get_sigma_bar(self) -> Optional[Tuple[float, float, float]]:
        """获取第一行的 sigma_bar (x, y, z)，若无则返回 None"""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT x, y, z FROM sigma_bar WHERE id = 1")
            result = cursor.fetchone()
            if result:
            # 将JSON字符串转换回向量
                return (
                    json.loads(result[0]) if result[0] else None,
                    json.loads(result[1]) if result[1] else None,
                    json.loads(result[2]) if result[2] else None
                )
            return None

    def update_sigma_bar(self, x: float, y: float, z: float) -> None:
        """更新或插入第一行的 sigma_bar"""
        
        x_json = json.dumps(x.tolist() if isinstance(x, np.ndarray) else x)
        y_json = json.dumps(y.tolist() if isinstance(y, np.ndarray) else y)
        z_json = json.dumps(z.tolist() if isinstance(z, np.ndarray) else z)
        with self._get_cursor() as cursor:
            # 检查是否存在 id=1 的行
            cursor.execute("SELECT id FROM sigma_bar WHERE id = 1")
            if cursor.fetchone():
                cursor.execute(
                    "UPDATE sigma_bar SET x=%s, y=%s, z=%s WHERE id=1",
                    (x_json, y_json, z_json)
                )
            else:
                cursor.execute(
                    "INSERT INTO sigma_bar (id, x, y, z) VALUES (1, %s, %s, %s)",
                    (x_json, y_json, z_json)
                )

    def insert_di_record(self, serial_no: str, x: float, y: float, z: float, 
                     is_anomaly_x: bool = False, is_anomaly_y: bool = False, is_anomaly_z: bool = False) -> None:
        """插入一个新的 d_i 记录，从第二行开始累积"""
            # 确定状态
        status = 'normal'
        if is_anomaly_x and is_anomaly_y and is_anomaly_z:
            status = 'anomaly'
        elif is_anomaly_x:
            status = 'anomaly'
        elif is_anomaly_y:
            status = 'anomaly'
        elif is_anomaly_z:
            status = 'anomaly'
        with self._get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO sigma_bar (serial_no, x, y, z, status) VALUES (%s, %s, %s, %s, %s)",
                (serial_no, str(x), str(y), str(z), status)
            )

    def get_all_di(self, status: str = None) -> List[Tuple]:
        """获取所有历史 d_i（不包括第一行和阈值记录），可按状态筛选"""
        with self._get_cursor() as cursor:
            if status:
                cursor.execute(
                    "SELECT id, serial_no, x, y, z, status FROM sigma_bar WHERE id > 1 AND status = %s AND status NOT LIKE 'threshold_%' ORDER BY id DESC",
                    (status,)
                )
            else:
                cursor.execute(
                    "SELECT id, serial_no, x, y, z, status FROM sigma_bar WHERE id > 1 AND status NOT LIKE 'threshold_%' ORDER BY id DESC"
                )
            return cursor.fetchall()


    def insert_anomaly_record(self, serial_no: str, x_anomaly: bool, y_anomaly: bool, z_anomaly: bool, 
                            x_threshold: float, y_threshold: float, z_threshold: float) -> None:
        """插入异常检测结果记录"""
        with self._get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO anomaly_detection 
                (serial_no, x_anomaly, y_anomaly, z_anomaly, x_threshold, y_threshold, z_threshold) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (serial_no, x_anomaly, y_anomaly, z_anomaly, x_threshold, y_threshold, z_threshold)
            )
    def get_anomaly_history(self, limit: int = 100) -> List[Tuple]:
        """获取最近的异常检测历史记录"""
        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT id, serial_no, x_anomaly, y_anomaly, z_anomaly, x_threshold, y_threshold, z_threshold, created_at FROM anomaly_detection ORDER BY id DESC LIMIT %s",
                (limit,)
            )
            return cursor.fetchall()
    def get_axis_anomaly_history(self, axis: str, limit: int = 100) -> List[Tuple]:
        """获取特定轴的异常检测历史记录"""
        column = f"{axis.lower()}_anomaly"
        with self._get_cursor() as cursor:
            cursor.execute(
                f"SELECT id, serial_no, {column}, {axis.lower()}_threshold, created_at FROM anomaly_detection ORDER BY id DESC LIMIT %s",
                (limit,)
            )
            return cursor.fetchall()

    def update_thresholds(self, x_threshold: Optional[float] = None, y_threshold: Optional[float] = None, z_threshold: Optional[float] = None) -> None:
        """更新异常检测阈值（所有轴）"""
        with self._get_cursor() as cursor:
            # 检查是否已存在阈值记录
            cursor.execute("SELECT id, x, y, z FROM sigma_bar WHERE status = 'threshold' LIMIT 1")
            result = cursor.fetchone()
            
            if result:
                # 获取现有值
                id, current_x, current_y, current_z = result
                
                # 只更新提供的值
                new_x = str(x_threshold) if x_threshold is not None else current_x
                new_y = str(y_threshold) if y_threshold is not None else current_y
                new_z = str(z_threshold) if z_threshold is not None else current_z
                
                # 更新现有记录
                cursor.execute(
                    "UPDATE sigma_bar SET x=%s, y=%s, z=%s WHERE id=%s",
                    (new_x, new_y, new_z, id)
                )
            else:
                # 插入新记录
                cursor.execute(
                    "INSERT INTO sigma_bar (serial_no, x, y, z, status) VALUES (%s, %s, %s, %s, %s)",
                    ("threshold", 
                    str(x_threshold) if x_threshold is not None else "0", 
                    str(y_threshold) if y_threshold is not None else "0", 
                    str(z_threshold) if z_threshold is not None else "0", 
                    "threshold")
                )



    def get_thresholds(self) -> Optional[Tuple[float, float, float]]:
        """获取异常检测阈值（所有轴）"""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT x, y, z FROM sigma_bar WHERE status = 'threshold' LIMIT 1")
            result = cursor.fetchone()
            
            if result:
                try:
                    return (
                        float(result[0]) if result[0] else None,
                        float(result[1]) if result[1] else None,
                        float(result[2]) if result[2] else None
                    )
                except (ValueError, TypeError):
                    return None
            
            return None
        
    def get_di_history(self, axis: str) -> List[float]:
        column = axis.lower()  # 直接使用轴的小写作为字段名（如X轴的di值存储在x字段）
        with self._get_cursor() as cursor:
            # 排除阈值记录和基准记录
            cursor.execute(
                f"SELECT {column} FROM sigma_bar WHERE id > 1 AND status NOT LIKE 'threshold%%' ORDER BY id DESC"
            )
            result = cursor.fetchall()
            history = []
            for row in result:
                if row[0] is None:
                    continue
                try:
                    # 将字符串转换为浮点数
                    di_value = float(row[0])
                    history.append(di_value)
                except (TypeError, ValueError) as e:
                    print(f"转换DI值失败: {row[0]}, 错误: {e}")
                    continue
            return history

    def get_di_history_by_axis(self, axis: str) -> List[float]:
        """按轴读取threshold记录中的DI历史数据"""
        column_map = {
            'X': 'x',
            'Y': 'y',
            'Z': 'z'
        }
        if axis.upper() not in column_map:
            raise ValueError(f"无效轴名: {axis}")
        
        column = column_map[axis.upper()]
        with self._get_cursor() as cursor:
            # 只读取状态为threshold的记录
            cursor.execute(
                f"SELECT {column} FROM sigma_bar WHERE status = 'threshold' ORDER BY id DESC"
            )
            history = []
            for row in cursor.fetchall():
                try:
                    # 解析JSON数组并展开为数值列表
                    di_values = json.loads(row[0])
                    if isinstance(di_values, list):
                        history.extend(di_values)
                except (json.JSONDecodeError, TypeError):
                    continue
            return history

    def get_current_thresholds(self) -> Dict[str, List[float]]:
        """获取各轴最新阈值（最后一条threshold记录）"""
        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT x, y, z FROM sigma_bar "
                "WHERE status = 'threshold' ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if not row:
                return {'X': [], 'Y': [], 'Z': []}
            
            thresholds = {}
            for i, axis in enumerate(['X', 'Y', 'Z']):
                try:
                    thresholds[axis] = json.loads(row[i])
                except (json.JSONDecodeError, TypeError):
                    thresholds[axis] = []
            return thresholds
        
    def get_all_di_history(self) -> List[Tuple[float, float, float]]:
        """
        获取所有历史 d_i（三轴）记录（不包括 id=1 的基准行和 status='threshold' 行）
        返回列表，元素格式为 (di_x, di_y, di_z)
        """
        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT x, y, z 
                  FROM sigma_bar 
                 WHERE id > 1 
                   AND status NOT IN ('threshold') 
                 ORDER BY id ASC
                """
            )
            rows = cursor.fetchall()
        history = []
        for x_str, y_str, z_str in rows:
            try:
                x = float(x_str)
                y = float(y_str)
                z = float(z_str)
                history.append((x, y, z))
            except (TypeError, ValueError):
                # 如果数据保存为 JSON 数组或字符串，可以在这里做额外解析
                continue
        return history