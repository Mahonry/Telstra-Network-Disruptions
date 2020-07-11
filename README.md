# Telstra-Network-Disruptions

Este repositorio contiene mi contribucion a a la competencia en en Kaggle [**Telstra Network Disruptions**](https://www.kaggle.com/c/telstra-recruiting-network/overview), cuyo objetivo es predecir la severidad de las interrupciones de su servicio, usando un conjunto de datos de características de sus registros de servicio, tiene la tarea de predecir si una interrupción es una falla momentánea o una interrupción total de la conectividad.

Expondre todo el proceso, desde el entendimiento de los datos hasta mis resultados finales.

## Entendimiento del negocio
  
Telstra es una empresa de telecomunicaciones con ubicacion en Australia, es el mayor proveedor de telefonía local y larga distancia, de servicios de telefonía móvil, acceso  telefónico, conexiones inalámbricas, DSL y de acceso a Internet por cable en Australia.

Como en cualquier empresa que involucre infraestrutura computacional, mantenimiento, crecimiento en la capacidad de brindar sus servicios y tecnologia de punta, entre otras caracteristicas, suelen presentarse fallas, estas pueden ser desde ligeras hasta muy severas, al ser una empresa dedicada a las telecomunicaciones en un mundo donde la conectividad es imprescindible, tener estas fallas identificadas y sus posibles causas, es vital para repararlas en el menor tiempo posible y mantener la calidad en los servicios.

Diversas causas indiviudales o una combinacion de estas, pueden llevar a un mismo tipo de falla o serveridad en la interrupcion del servicio, teniendo las fallas clasificadas, el problema a resolver es: ¿Que factores causan cada tipo de falla? para asi poder prevenirlas.

Para esto Telstra a pone a disposicion del publico una simulacion de los datos que se generan al producirse un tipo de falla, con el objetivo de crear un modelo predictivo clasificatorio.


## Comprension de los datos

El objetivo del problema es predecir la gravedad de la falla de la red de Telstra a la vez en una ubicacion particular en funcion de los datos de registro disponibles. Los datos que se proporcionan vienen segmentados en diferentes datasets, que se unen entre si por medio de un 'id', :

  - train.csv:  Set de datos para entrenar al algortimo, contiene tres columnas.

  - test.csv: Set de datos para realizar las predicciones y subir los resultados, contiene dos columnas.
  
  - event_type.csv:  Tipo de evento relacionado con el conjunto de datos principal.

  - log_feature.csv: características extraídas de los archivos de registro.

  - resource_type.csv: tipo de recurso relacionado con el conjunto de datos principal.

  - severity_type.csv: tipo de gravedad de un mensaje de advertencia procedente del registro.

## Insepeccion los datos.

### train.csv

    Tamaño: (7381, 3) 
    
    7831 registros para entrenar el algoritmo.

    Columnas:
        - id: Identificador unico por falla.
        - location: Locacion donde se dio la falla. (Variable Categorica)
        - fault_severity: Clasificación de la falla. 
    
    7831 Valores unicos en id
    929 Valores unicos en location
    
    Distribucion por fault_severity:

    0    64.81%
    1    25.34%
    2     9.83%

    Muestra de la distribucion por location:

    location 821     9.149623%
    location 1107    8.396125%
    location 734     8.073197%
    location 1008    7.642626%
    location 126     7.642626%

- **Por la distribucion de los datos nos damos cuenta que tenemos un problema de clasificacion multiclase, pero que estan desbalanceadas, ya que solo el 9.8 % de los registros pertenecen a una falla del tipo 2.**

- El dataset no contiene mas informacion, las caracteristicas vendran dadas al hacer las uniones con los demas datasets.

- Sin valores nulos

### test.csv

    Tamaño : (11171, 2)

    Columnas:
        - id: Identificador unico por falla.
        - location: Locacion donde se dio la falla.

    11171 Valores unicos en id.
    1039 Valores unicos por location.

    Muestra de la distribucion por location:

    location 734     10.490857
    location 653     10.394610
    location 126     10.202117
    location 1107    10.202117
    location 810      9.817132
    
    Total data (train + test):   18552

- Sin valores nulos.

### even_type.csv

    Tamaño (31170, 2)

    Columnas:
        - id: Identificador unico por falla.
        - event_type: Tipo de evento (Variable categorica)
    
    53 Valores unicos en event_type
    18552 Valores unicos en id

    Muestra de la distribucion:

    event_type 11    25.306384%
    event_type 35    21.222329%
    event_type 34    19.015079%
    event_type 15    14.100096%
    event_type 20     4.677575%

- Contiene el mismo numero de id's que Total data (train + test), sin embargo el numero de registros totales es 311170 lo que seguiere que hay mas de un evento por id.

- Sin valores nulos.

### log_feature.csv

    Tamaño (58671, 3)

    Columnas: 
        - id: Identificador unico por falla.
        - log_feature: caracteristica sin mayor descripcion (Variable categorica)
        - volume: caracteristica sin mayor descripcion (Variable numerica)

    18552 Valores unicos en id 
    386 Valores unicos en log_feature
    341 Valores unicos en volume

    Muestra de la distribucion en log_feature:

    feature 312    8.977178%
    feature 232    8.102811%
    feature 82     5.917745%
    feature 203    4.811576%
    feature 313    3.655980%

- Contiene el mismo numero de id's que Total data (train + test), sin embargo el numero de registros totales es 58671 lo que seguiere que hay mas de un evento por id.

- Sin valores nulos

### resource_type.csv

    Tamaño (21076, 2)

    Columnas:
        - id: Identificador unico por falla.
        - resource_type: Tipo de recurso (Variable categorica)
    
    10 Valores unicos en resource_type
    18552 Valores unicos en id

    Muestra de la distribucion:

    resource_type 4      1.565762%
    resource_type 7      2.362877%
    resource_type 6      2.761435%
    resource_type 2     42.313532%
    resource_type 8     48.718922%

- Contiene el mismo numero de id's que Total data (train + test), sin embargo el numero de registros totales es 21076 lo que seguiere que hay mas de un recurso por id.

- Sin valores nulos

### severity_type.csv

    tamaño: (18552, 2)

    Columnas:
        - id: Identificador unico por falla.
        - severity_type: A menudo, este es un tipo de gravedad de un mensaje de advertencia proveniente del registro. (Variable categorica)
    
    5 Valores unicos en severity_type
    18552 valores unicos en id

    Distribucion:

    severity_type 1    47.046141%
    severity_type 2    47.094653%
    severity_type 3     0.043122%
    severity_type 4     5.465718%
    severity_type 5     0.350367%

- Contiene el mismo numero de id's que Total data (test + train ) y contiene 18552 registros lo que sugiere que se asocia solo un tipo de 'severity_type' por id.

- Sin valores nulos


## Preparacion de los datos.

Todo el proceso de preparacion de los datos puede verse en el notebook [Data_Preparation](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Data_Preparation.ipynb).

Los datasets generados pueden verse en la carpeta [Data extraida](https://github.com/Mahonry/Telstra-Network-Disruptions/tree/master/Data%20extraida).

Se generaron 3 datasets con diferentes, para probar su desempeño en los modelos, los 3 se crearon de manera similar, a continuacion explicare los pasos comunes para consolidarlos:

1._ Para los datasets, 'even_type.csv','resource_type.csv' y 'severity_type.csv.' se agrupa las columnas  de variables categoricas por 'id'.

2._ De nuevo para los datasets, even_type.csv','resource_type.csv' y 'severity_type.csv.' una vez agrupadas las variables categoricas, se aplica el metodo get_dummies() de la libreria pandas, para obtener una columna con cada variable y el conteo de cuantas veces se repitio esa variable en el id.

3._ Se agrega una columna con la suma del total de variables que se contablizaron para ese 'id'

4._ Se cuentan las frecuencias de las locaciones para los dataset 'train.csv' y 'test.csv' se agrega una columna con el total.

5._ Se convierten los datos de las locaciones de los datasets 'train.csv' y 'test.csv' a numericos por medio de la funcion label_enconder().

6._ Se consolidan los datasets creando merge por id con los datasets 'train.csv' y 'test.csv'.

El procedimiento difiere para los datasets en lo siguiente:

- En los datasets ['train_1'](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Data%20extraida/train_1.csv) y ['test_1'](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Data%20extraida/test_1.csv) Se utilizo log['Features_bin'] = pd.qcut(log.log_feature, 10) agrupando la variable categorica 'log_feature' en 10 grupos percentiles, el numero total de features en estos datasets es de 84.

- En los datasets ['train_2'](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Data%20extraida/train_2.csv) y ['test_2'](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Data%20extraida/test_2.csv) Se utilizo log['Features_bin'] = pd.cut(log.log_feature, 193) para reducir a la mitad la cantidad de features dadas por 'log_feature',
el numero total de features en estos datasets es de 269.

- En los datasets ['train_consolidado_completo'](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Data%20extraida/train_consolidado_complete.csv) y ['test_consolidado_completo'](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Data%20extraida/test_consolidado_complete.csv) se aplico el metodo get_dummies() a todas las features incluidas las locaciones y 'log_feature', el numero total de features en este dataset es de 1390.


## Modelado y Evaluacion

Reliace varios modelos y pruebas, ya que con el primer prototipo de modelo, obtuve un puntaje en Kaggle que cumplia con el requisito de ser menor a 0.7 y me puse como meta personal bajarlo lo mas posible mientras tuviera tiempo, por lo describire como llegue a obtener mi mejor puntaje que fue _____________, y como evalue los modelos entrenados.

**Consideraciones importantes**

- Me concentre en evaluar y mejorar los modelos con la metrica 'multi-class logaritmic loss' ya que seria esta la que Kaggle usaria para puntuar mis predicciones.

-  Ya que era un problema con clases desequilibradas habia que prestar especial atencion a estas clases  y usar una metrica correcta, para los modelos finales, tambien me fije en el f1_score y en las matriz de confusion.


### [Primer prototipaje](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Primer_prototipaje.ipynb)

En este notebook, relice las primeras pruebas, para ver el rendimiento de disitintos algoritmos con los distintos datsets, no me centre tanto en el rendimiento, si no en tener un panorama general, para luego optimizarlo.

Seleccione cuatro modelos que en general sirvieran para clasificaciones multiclase y con los que ademas ya estuviera un poco familiarizado con ellos, para poder entender mejor lo que hacian, los algoritmos que seleccione fueron:

* Random Forest Classifier (RF)

* Gradient Boosting Classifier (GBC)

* Desicion Tree Classifier (DT)

* K Nearest Neighbors Classifier (KNN) 

En estas pruebas no realice ninguna optimizacion de hiperparametros por lo que, todos los modelos fueron entrandos con los parametros estandar.

Se pusieron a prueba los datasets 'train_1' y 'train_consolidado_completo', sin embargo, al hacer las primeras pruebas, me percate que realmente no habia una diferencia significativa en los rendimientos y el tiempo de entrenamiento del dataset 'train_consolidado_completo' era signficativamente mayor, como solo queria ver el panorama general de los rendimiento, descarte este ultimo dataset y todas las pruebas las realice con 'train_1'.

#### Primera prueba, dataset sin modificaciones

    - Dividi los datos de 'train_1' en X_train, X_test, y_train, y_test con la proporcion 70:30.

    - Entrene los modelos con X_train y y_train, predije con X_test y saque las metricas 'log_loss' y el score default de los modelos que en todos es 'mean accuracy'.

    Los resultados fueron los siguientes:

    Model: RF   log_loss: 0.8315947745024285
    Model: RF   score: 0.7051918735891648
    ___________________________________________
    Model: GBC   log_loss: 0.5976638554985724
    Model: GBC   score: 0.7187358916478556
    ____________________________________________
    Model: DT   log_loss: 10.876911920615587
    Model: DT   score: 0.6785553047404064
    ____________________________________________
    Model: KNN   log_loss: 2.459482808226647
    Model: KNN   score: 0.6397291196388262

    Observamos que en una primera aproximación sin modifica las features 
    o realizar seleccion de componentes principales los modelos con mejor 
    performance son: Random Forest y Gradient Bossting Classifier, 
    siendo este último el que obtuvo la menor perdida.

#### Segunda prueba, rescanlando los datos

    - Realice los mismos pasos que en el punto anterior, pero esta vez rescale los datos en train, con la funcion StandardScaler.

    Los resultados fueron los siguientes:

    Model: RF   log_loss: 0.7614260921343133
    Model: RF   score: 0.709255079006772
    ________________________________________
    Model: GBC   log_loss: 0.5976547758516163
    Model: GBC   score: 0.7182844243792325
    ________________________________________
    Model: DT   log_loss: 10.798320418883904
    Model: DT   score: 0.6808126410835215
    ________________________________________
    Model: KNN   log_loss: 2.9901517056875164
    Model: KNN   score: 0.6735891647855531
    ________________________________________

    Vemos nuevamente que la menor perdida la tuvo el Gradien Boosting Classifier, 
    sin embargo, no hubo un desenso signficativo en comparacion a la iteracion anterior

#### Analizis de features

Una caracteristica de los algortimos Random Forest Classifier y Gradient Boosting Classfier es que devuelve las relevancia de features, por lo que se me hizo intersante analizar que features habian sido mas relevantes para el Gradient Boosting Classfier y comparar con una tecnica de seleccion de features automatica SelectKBest''.

- Visualice las relevancia de features para el Gradient Boosting Classfier.
[imagen]

- Entrene el selector automatico 'SelectKBest' con k = 64 y las visualice.
[imagen]

- Compare si estos dos mecanismos habian elegido las mismas features relevantes, coincidieron en 10 features.

##### Tercera prueba, Seleccion automatica de features y rescalamiento de datos

    - De igual manera reescale los datos, pero use el selector automatico para que seleccionara las features, los resultados son los siguientes:

    Model: RF   log_loss: 0.7117107542202854
    Model: RF   score: 0.7033860045146727
    ________________________________________
    Model: GBC   log_loss: 0.5975829761383757
    Model: GBC   score: 0.7200902934537246
    ________________________________________
    Model: DT   log_loss: 10.844786866741318
    Model: DT   score: 0.6799097065462754
    ________________________________________
    Model: KNN   log_loss: 2.9273783436843583
    Model: KNN   score: 0.6762979683972912
    ________________________________________

#### Analizando rendimientos

Una vez obtenidos todos estos rendimientos, seleccione el modelo que tuviera la menor medida en la metrica 'log_loss' que fue Gradient Boosting Classifier y además este mismo tenia el mejor score, por lo que seleccione este modelo para optimizarlo, sin embargo con estas condiciones, realice una primera prediccion ['Submission_1'](https://github.com/Mahonry/Telstra-Network-Disruptions/blob/master/Submission/submision_1.csv) para puntuar en Kaggle.

El resultado fue:

[imagen]
