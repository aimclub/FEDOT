from launch.utils import *


if __name__ == '__main__':
    path = 'med_data.xlsx'
    dtype_dic = {'Вид поражения: очаг – 1, диффузная зона -2': str,
                 'Количество: единичные – 1, множественные – 2': str,
                 'degree': str, 'peri': str, 'talamus': str, 'hypotalamus': str,
                 'medium brain': str, 'mozol telo': str}
    df = pd.read_excel(path, engine='openpyxl', dtype=dtype_dic)
    # ВАЖНО! В конце документа присутствуют пустые строки - исключим их
    df = df.head(52)

    # Выберем только нужные нам признаки
    features_plus_target = ['CRS-R- index 1 при поступлении',
                            'CRS-балл-1-  при поступлении',
                            'Вид поражения: очаг – 1, диффузная зона -2',
                            'Количество: единичные – 1, множественные – 2',
                            'degree', 'peri', 'talamus', 'hypotalamus',
                            'medium brain', 'mozol telo']
    features = features_plus_target[2:]
    df = df[features_plus_target]
    # Удалим пропуски
    df = df.dropna()

