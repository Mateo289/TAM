# Cómo montar un entorno virtual en python para ejecutar el script best.tochscript

1. Tener instalado pip en su equipo

Para verificar que tenga instalado pip en su sistema.

```bash
pip --version
```

Si no lo tiene instalado, instale pip (Viene con python).
Tiene que tener instalado python.

2. Con pip instalado, hay que instalar pipenv que es un gestor de dependencias.

```bash
pip install pipenv
```

3. Con pipenv instalado podemos crear un ENTORNO VIRTUAL, para poder correr el archivo YoloViewer.py.

    3.1 Abrimos una terminal en la ruta donde tenemos el archivo YoloViewer.py

    3.2 Ejecutamos el entorno virtual

        ```bash
        pipenv shell
        ```
    
    3.3 Ahora podemos instalar las librerías que necesitamos

         ```bash
         pipenv install torch
         ```

         ```bash
         pipenv install opencv-python
         ```
    
    3.4 Con las librerías instaladas corremos el YoloViewer.python

    ```bash
    python YoloViewer.py
    ```