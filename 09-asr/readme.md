# ASR

По классике, для многопоточки под виндой нужно, чтобы каждый процесс имел представление об исполняемых функциях, для этого они должны быть определены в `.py`, тогда и в ноутбуке можно будет всё исполнить - будь то mpire, будь то multiprocessing. Ждать результата, чтобы убедиться исправлена ли ошибка, в один момент стало слишком долго и я отказался от ноутбуков на 99%

В `metrics.py` лежат метрики. Обратите внимание, что вариант [CER из torchmetrics](https://torchmetrics.readthedocs.io/en/stable/text/char_error_rate.html) по сути **уже** является relative, поэтому спокойно считая с помощью [библиотеки](https://pypi.org/project/python-Levenshtein/) число замен/вставок/удалений достаточно не поделить на длину и получить правильное значение метрики. Я долго не замечал этого факта и откатился до вычисления расстояния Левенштейна через матрицу по формуле, чтобы контролировать на каждом шаге происходящее и таки найти ошибку

С argmax декодингом проблем быть не может, но не с бимсёрчем. Его вариантов на существует не так уж и мало, кроме предложенного в [First-Pass Large Vocabulary Continuous Speech
Recognition using Bi-Directional Recurrent DNNs](http://arxiv.org/abs/1408.2873v2), можно найти [реализацию](https://github.com/githubharald/CTCDecoder/blob/master/ctc_decoder/beam_search.py) с применением языковой модели и много-много других. Я перепробовал множество версий, они все похожи в своих идеях, отличия только в том, что подразумевается под итоговой вероятностью бима и как её считать с учётом всех специальных токенов, среди которых особое внимание стоит уделить blank - на нём завязано многое. Тем не менее, никакая из них не дала результата, чтобы top1 cer/wer были лучше argmax

Последней [версией](ctc_decoder.py) было что-то похожее (или уже даже в точности) на указанное по ссылке, с применением простейшей языковой модели, но результат совершенно аналогичный

Поскольку самой модели, генерящей вероятности токенов, нет, а есть только её результат, я предположил, что "проблема" в ней, т.е. генерируемые ей вероятности для, возможно, похоже звучащих токенов, не близки друг к другу, поэтому и бимсерчем найти более правильный результат декодирования невозможно. В бимах есть такой "более правильный" результат, поскольку oracle падает, но их вероятность не top1. Проверить это явно возможности нет, но, по идее, больше грешить не на что

> хотя, кажется, проверить можно - нужно посмотреть явно на вероятности тех токенов, на которых допускаются ошибки и если там для похоже звучащих токенов вероятности не похожи, то гипотеза подтвердится

В `process.py` остался расчёт только для 4 бимов, в [ноутбуке](asr.ipynb) графики построены для `[4, 8, 16, 32]`, но не той версией бим серча, которая в `ctc_decoder.py`, однако поскольку они похожи, разницы в целом нет