import copy
import datetime

import pandas as pd

from config import cn_en_dict
from utils import thread_pool
from utils.uploader import Uploader


class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.raw_data = copy.deepcopy(data)
        self.must_drop_col = ['Frame(Long)_Qty','Fram(Shot)_Qty']
        # 三个日期 SN_Generated_Time
        # Packaging_Time 打包日期
        # Stocking_Time 入库日期
        self.time_columns = [col for col in self.data.columns if 'Time' in col]

    def reset_data(self):
        del self.data
        self.data = copy.deepcopy(self.raw_data)

    def run_drop_na(self):
        self.data.fillna(axis=0, inplace=True)
        return self

    def build(self):
        return self.data

    def auto_build(self, drop_na=True):
        builder = DataProcessor(self.data)

        # 使用反射来找到所有以 'run_' 开头的方法
        for method_name in dir(DataProcessor):
            if method_name.startswith('run_'):
                method = getattr(builder, method_name)
                if callable(method):
                    method()  # 调用方法
        if drop_na:
            return builder.drop_na().build()
        else:
            return builder.build()

    def read_raw_data(self, path):
        self.data = pd.read_parquet(path)
        return self

    def gen_y(self):
        # 构建Y列
        self.data['扇形损坏'] = self.data['damaged_detail'].apply(lambda x: 1 if x == 1 else 0)
        self.data['不规则损坏'] = self.data['damaged_detail'].apply(lambda x: 1 if x == 2 else 0)
        return self

    def drop_date_col(self):
        """日期列"""
        time_columns = self.time_columns
        self.data = self.data.drop(time_columns, axis=1)
        return self

    def union_name(self):
        """"类别列的名称清洗统一"""
        # 背玻供应商替换字典
        back_sheet_dict = {'锡海达': '无锡海达', '华美': '常州华美', '海达': '无锡海达', '信义': '信义光能', }

        replace_list = {'背玻供应商': back_sheet_dict}

        for k, v in replace_list.items():
            self.data[k] = self.data[k].replace(v)
        return self

    def drop_col(self):
        """去除列"""
        drop_col = ['Backsheet_Type', '区域', '国家', 'Backsheet_Batch(Mes)', 'Glass2_Batch', 'J-box_Barcode',
                    'B-Glue_Material_No', 'Frame(long-side)_Material_No.', 'Frame(Long)_Qty', 'Fram(Shot)_Qty',
                    'Module_Material_No.', 'Module_Grade', 'Backsheet_Supplier', 'Backsheet_Material_No.',
                    'ARC-glass_Material_No.', 'Glass2_Supplier', 'Glass2_Material_No', 'Glass2_Material_Desc',
                    'Glass2_Batch(Mes)', 'J-box_Description', 'Jbox_Batch(Mes)', 'B-Glue_Batch(Mes)',
                    'EVA_Material_Desc',
                    'EVA_Batch(Mes)', 'Frame(Long)_Batch(Mes)', 'Frame(short-side)_Material_No.',
                    'Frame(Short)_Batch(Mes)',
                    'Cell_Efficiency', 'Cell_Material_No.', 'Cell_Batch(Mes)', 'Frame_Silicon_Material_No',
                    'Frame_Silicon_Batch(Mes)', 'Jbox_Pedestal_Glue_Material_No', 'Jbox_Pedestal_Glue_Batch(Mes)',
                    'Pallet_No.',
                    'Lot_No.',
                    'Power', 'rowOrder', 'ARC-glass_Supplier', 'ARC-glass_Material_Desc', 'ARC-glass_Batch',
                    'ARC-glass_Batch(Mes)', 'A-Glue_Batch(Mes)', 'CleanEVA_Batch(Mes)', '背玻批次号', 'AG_Desc',
                    'Clean_EVA_Material_Desc']

        cat_col2_high_dim = ['Backsheet_Batch', 'J-box_Batch', 'A-Glue_Batch', 'B-Glue_Batch', 'EVA_Batch',
                             'Clean_EVA_Batch',
                             'Farme(long-side)_Batch', 'Frame(short-side)_Batch', 'Cell_Grade', 'Cell_Material_Desc',
                             'Cell_Batch', 'Frame_Silicon_Batch', 'Jbox_Pedestal_Glue_Batch']

        self.data = self.data.drop(cat_col2_high_dim, axis=1)
        self.data = self.data.drop(drop_col, axis=1)

        return self

    def drop_must(self):
        self.data = self.data.drop(self.must_drop_col, axis=1)

    def type_find(self):
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_numeric(self.data[col])
                except ValueError:
                    pass

    def type_change(self):
        cat_col = ['项目地', '背玻供应商', 'Module_Type', 'Module_Material_Desc', 'EL_Grade', 'Color_Grade',
                   'Current_Grade', 'String_Connector_Supplier',
                   'Work_Order', 'Factory', 'Workshop', 'Backsheet_Material_Desc', 'Backsheet_Thickness',
                   'J-box_Supplier',
                   'J-box_Material_No.', 'Diode_Type', 'Connector_Type', 'AG_Supplier', 'A-Glue_Material_No',
                   'BG_Supplier',
                   'BG_Desc', 'EVA_Supplier', 'EVA_Material_No.', 'EVA_Type', 'EVA_Thickness', 'Clean_EVA_Supplier',
                   'Clean_EVA_Material_No.', 'Frame(long-side)_Supplier', 'Frame_Size', 'Frame(long-side)_Desc',
                   'Frame(short-side)_Supplier', 'Frame(short-side)_Desc', 'Cell_Supplier', 'Cell_Type',
                   'Cell_Connector_Supplier', 'Cell_Connector_Desc', 'Cell_Connector_Desc', 'String_Connector_Desc',
                   'Frame_Silicon_Supplier', 'frame_Silicon_Desc', 'J-box_Pedest_Supplier', 'J-box_Pedest_Desc',
                   'Cable_Length']

        float_col = ['Isc(A)', 'Voc(V)', 'Pmax(W)', 'Vpm(V)', 'Ipm(A)', 'FF', 'A-Glue_Qty', 'B-Glue_Qty', 'EVA_Qty',
                   'CleanEVA_Qty', 'Cell_Qty', 'Jbox_Pedestal_Glue_Qty', 'Frame_Silicon_Qty']

        drop_cat_col = ['Backsheet_Type', '区域', '国家', 'Backsheet_Batch(Mes)', 'Glass2_Batch', 'J-box_Barcode',
                        'B-Glue_Material_No', 'Frame(long-side)_Material_No.', 'Frame(Long)_Qty', 'Fram(Shot)_Qty',
                        'Module_Material_No.', 'Module_Grade', 'Backsheet_Supplier', 'Backsheet_Material_No.',
                        'ARC-glass_Material_No.', 'Glass2_Supplier', 'Glass2_Material_No', 'Glass2_Material_Desc',
                        'Glass2_Batch(Mes)', 'J-box_Description', 'Jbox_Batch(Mes)', 'B-Glue_Batch(Mes)',
                        'EVA_Material_Desc',
                        'EVA_Batch(Mes)', 'Frame(Long)_Batch(Mes)', 'Frame(short-side)_Material_No.',
                        'Frame(Short)_Batch(Mes)',
                        'Cell_Efficiency', 'Cell_Material_No.', 'Cell_Batch(Mes)', 'Frame_Silicon_Material_No',
                        'Frame_Silicon_Batch(Mes)', 'Jbox_Pedestal_Glue_Material_No', 'Jbox_Pedestal_Glue_Batch(Mes)',
                        'Pallet_No.', 'Lot_No.', 'ARC-glass_Supplier', 'ARC-glass_Material_Desc', 'ARC-glass_Batch',
                        'ARC-glass_Batch(Mes)', 'A-Glue_Batch(Mes)', 'CleanEVA_Batch(Mes)', '背玻批次号', 'AG_Desc',
                        'Clean_EVA_Material_Desc']

        drop_num_col = ['Power', 'rowOrder']

        time_columns = [col for col in self.data.columns if 'Time' in col]

        obj_col = ['SN']

        cat_col2_high_dim = ['Backsheet_Batch', 'J-box_Batch', 'A-Glue_Batch', 'B-Glue_Batch', 'EVA_Batch',
                             'Clean_EVA_Batch',
                             'Farme(long-side)_Batch', 'Frame(short-side)_Batch', 'Cell_Grade', 'Cell_Material_Desc',
                             'Cell_Batch', 'Frame_Silicon_Batch', 'Jbox_Pedestal_Glue_Batch']

        y_col = ['is_damaged', 'damaged_detail', '扇形损坏', '不规则损坏']
        final_y = [col for col in y_col if col in self.data.columns]
        category = cat_col + drop_cat_col + cat_col2_high_dim
        num = float_col + drop_num_col
        for col in category:
            self.data[col] = self.data[col].astype('category')
        for col in num:
            self.data[col] = self.data[col].astype('float32')
        # for col in drop_num_col:
        #     self.data[col] = self.data[col].astype('int32')
        for col in time_columns:
            self.data[col] = pd.to_datetime(self.data.loc[:, 'Stocking_Time'], format='ISO8601').dt.date
            self.data[col] = self.data[col].astype('datetime64[ns]')
            #     self.data[col] = pd.to_datetime(self.data[col], format='%Y-%m-%d %H:%M:%S').dt.date
            # except ValueError:
            #     self.data[col] = pd.to_datetime(self.data[col], format='%Y-%m-%d').dt.date
            # self.data[col] = pd.to_datetime(self.data[col], format='%Y-%m-%d %H:%M:%S')
        for col in final_y:
            self.data[col] = self.data[col].astype('int8')

        # 避免出现不在columns中的列
        # cat_col = [col for col in cat_col if col in self.data.columns]
        # num_col = [col for col in num_col if col in self.data.columns]
        # self.data[cat_col] = self.data[cat_col].apply(lambda x: x.astype('category'))
        # self.data[num_col] = self.data[num_col].apply(lambda x: x.astype('float32'))

        return self

    def gen_multi_index(self):
        multi_index = pd.MultiIndex.from_tuples([(k, cn_en_dict.col_names.get(k, '')) for k in self.data.columns])

        # 将多级列名赋值给df
        self.data.columns = multi_index

        return self

    def sort_column(self):
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)
        return self

    @staticmethod
    def is_multi_level_columns(df):
        # 获取第一个列名
        first_column = df.columns[0]
        # 检查第一个列名是否是元组
        return isinstance(first_column, tuple)

    @staticmethod
    def upload_task(df, file):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        file_name = formatted_datetime + '.parquet'
        print(f'filename{file_name}')
        Uploader().upload_bytes(file, file_name)

    def thread_upload(self):
        future = thread_pool.ThreadPool().submit(self.upload_task, 'file_name')
