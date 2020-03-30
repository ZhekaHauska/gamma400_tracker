# gamma400_tracker
Обработка данных конвертер-трекера телескопа ГАММА-400.

В папке `project_manager` находится программа реализующая запуск программы моделирования данных на сервере с PBS, учёт успешо смоделированных событий и остановку процесса моделирования в случае избытка событий с нужными параметрами. Перечень параметров событий и другие параметры моделирования указываются в файле `input.yaml`.

В папке `track-recognition` расположены программы непосредственно выполняющие обработку смоделированных данных.  
`TrackGen.py` описывает класс, позволяющий выбрать и подготовить необходимое колличество событий из базы данных, создать генераторы, которые используются для обучения и тестирования нейронных сетей.  
В файле `tracker.py` описана нейронная сеть для распознавания треков электрон позитронной пары.
