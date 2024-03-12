FEDOT и Docker
============

Здесь представлены Docker файлы для запуска FEDOT


Версии
=========

- **Dockerfile** Полная версия FEDOT (fedot + fedot[extra]) для python 3.8
- **Dockerfile_light** Лёгкая версия FEDOT для python 3.8
- **GPU** Версия с поддержкой GPU для python 3.8
- **Dockerfile_Jupiter** Версия с Jupiter notebook для python 3.10. Ниже есть описание запуска для Linux.


Jupiter
=========
- **Проверте наличе docker** Docker должен быть установлен
- **git clone** получаем файлы из git
- **cd FEDOT** переходим в папку проекта
- **cd docker** переходим в папку с Docker файлами
- **docker build -t jupyter-fedot-tst -f Dockerfile_Jupiter .** строим образ и называем его "jupyter-fedot"
- **docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter-with-fedot** запускаем, будет доступен по URL http://[YOUR_IP]:8888, храним все файлы в текущей папке
- **копируем ссылку с ключем и открываем** - если запуск прошел как ожидается будет отображена ссылка следующего вида ..  http://127.0.0.1:8888/lab?token=db8ce02fbed23c3ecd896408a494de176a70d73cf51e203f