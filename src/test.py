import hydra
from hydra import compose, initialize
import os
from typing import Tuple
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
import dvc.api

def read_datastore() -> Tuple[pd.DataFrame, str]:
    initialize(config_path="../configs", job_name="extract_data", version_base=None)
    cfg = compose(config_name="main")
    print("PATH", os.path.join(cfg.paths.root_path, 'data', 'samples', cfg.datasets.sample_filename))
    print("REPO", cfg.paths.root_path)
    url = dvc.api.get_url(
        path=os.path.join('data', 'samples', cfg.datasets.sample_filename),
        repo=os.path.join(cfg.paths.root_path),
        rev=str(cfg.datasets.version),
        remote=cfg.datasets.remote,
    )

    data = pd.read_csv(url)

    return data, str(cfg.datasets.version)

def dup():
    array = ['price', 'item_seq_number', 'image_top_1', 'deal_probability',
            'params_length', 'region_Башкортостан', 'region_Белгородская область', 'region_Владимирская область',
            'region_Волгоградская область', 'region_Воронежская область', 'region_Иркутская область', 'region_Калининградская область',
            'region_Кемеровская область', 'region_Краснодарский край', 'region_Красноярский край', 'region_Нижегородская область', 
            'region_Новосибирская область', 'region_Омская область', 'region_Оренбургская область', 'region_Пермский край', 
            'region_Ростовская область', 'region_Самарская область', 'region_Саратовская область', 'region_Свердловская область', 
            'region_Ставропольский край', 'region_Татарстан', 'region_Тульская область', 'region_Тюменская область',
            'region_Удмуртия', 'region_Ханты-Мансийский АО', 'region_Челябинская область', 'region_Ярославская область',
            'category_name_Автомобили', 'category_name_Бытовая техника', 'category_name_Детская одежда и обувь',
            'category_name_Дома, дачи, коттеджи', 'category_name_Квартиры', 'category_name_Мебель и интерьер',
            'category_name_Одежда, обувь, аксессуары', 'category_name_Предложение услуг', 'category_name_Ремонт и строительство',
            'category_name_Телефоны', 'category_name_Товары для детей и игрушки', 'city_Ангарск', 'city_Барнаул', 'city_Белгород',
            'city_Бийск', 'city_Владимир', 'city_Волгоград', 'city_Волжский', 'city_Воронеж', 'city_Дзержинск', 'city_Екатеринбург',
            'city_Ижевск', 'city_Иркутск', 'city_Казань', 'city_Калининград', 'city_Кемерово', 'city_Краснодар', 'city_Красноярск',
            'city_Магнитогорск', 'city_Набережные Челны', 'city_Нижневартовск', 'city_Нижний Новгород', 'city_Нижний Тагил',
            'city_Новокузнецк', 'city_Новороссийск', 'city_Новосибирск', 'city_Омск', 'city_Оренбург', 'city_Пермь',
            'city_Ростов-на-Дону', 'city_Самара', 'city_Саратов', 'city_Сочи', 'city_Ставрополь', 'city_Стерлитамак', 'city_Сургут',
            'city_Таганрог', 'city_Тольятти', 'city_Тула', 'city_Тюмень', 'city_Уфа', 'city_Челябинск', 'city_Энгельс',
            'city_Ярославль', 'parent_category_name_Для бизнеса', 'parent_category_name_Для дома и дачи',
            'parent_category_name_Животные', 'parent_category_name_Личные вещи', 'parent_category_name_Недвижимость',
            'parent_category_name_Транспорт', 'parent_category_name_Услуги', 'parent_category_name_Хобби и отдых',
            'user_type_Private', 'user_type_Shop', 'param_1_Samsung', 'param_1_iPhone', 'param_1_missing', 'param_1_Аксессуары',
            'param_1_Детская мебель', 'param_1_Детские коляски', 'param_1_Для девочек', 'param_1_Для дома', 'param_1_Для кухни',
            'param_1_Для мальчиков', 'param_1_Другая', 'param_1_Другое', 'param_1_Женская одежда', 'param_1_Игрушки',
            'param_1_Инструменты', 'param_1_Книги', 'param_1_Комплектующие', 'param_1_Кровати, диваны и кресла', 'param_1_Мужская одежда', 
            'param_1_Приборы и аксессуары', 'param_1_Продам', 'param_1_Ремонт, строительство', 'param_1_С пробегом',
            'param_1_Сдам', 'param_1_Стройматериалы', 'param_1_Телевизоры и проекторы', 'param_1_Товары для кормления',
            'param_1_Транспорт, перевозки', 'param_1_Шкафы и комоды', 'param_2_missing', 'param_2_Брюки', 'param_2_Верхняя одежда',
            'param_2_Другое', 'param_2_Обувь', 'param_2_Платья и юбки', 'param_2_Трикотаж', 'param_3_42–44 (S)', 'param_3_44–46 (M)', 
            'param_3_46–48 (L)', 'param_3_74-80 см (7-12 мес)', 'param_3_86-92 см (1-2 года)', 'param_3_98-104 см (2-4 года)',
            'param_3_Other', 'param_3_missing', 'param_3_Вторичка', 'title_length', 'description_length', 'квартира', 'м²',
            'продавать', 'эта', 'весь', 'год', 'дом', 'квартира', 'комплект', 'новый', 'описание', 'отличный', 'очень', 'продавать',
            'размер', 'см', 'состояние', 'торг', 'хороший', 'цвет', 'цена', 'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos']
    
    def list_duplicates(seq):
        seen = set()
        seen_add = seen.add
        # adds all elements it doesn't know yet to seen and all other to seen_twice
        seen_twice = set( x for x in seq if x in seen or seen_add(x) )
        # turn the set into a list (as requested)
        return list( seen_twice )
    
    print(list_duplicates(array))


if __name__ == '__main__':
    dup()