### Ставим Python 310 (Не забудьте поставить галочку "add to PATH")
https://www.python.org/ftp/python/3.10.11/

### Ставим куду 118
https://developer.nvidia.com/cuda-11-8-0-download-archive

### Создать виртуальное окружение
`python -m venv venv`

### Активировать виртуальное окружение
Шиндоус:  
`venv\Scripts\activate.bat`  
Для PowerShell  
`venv\Scripts\Activate.ps1`  
Линукс:  
`source venv/bin/activate`

### Установить зависимости
`pip install -r requirements.txt`

### Установить зависимости Pytorch с CUDA
`pip install -r extensions/cuda/requirements.cuda.txt --index-url https://download.pytorch.org/whl/cu118`

### Установить расширение
`python setup.py install`  
Или переместить `extensions/cuda/build_lib/imp_cuda_extension-0.0.0-py3.10-win-amd64.egg` в `venv/Lib/site-packages`