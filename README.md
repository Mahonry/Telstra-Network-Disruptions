# Telstra-Network-Disruptions

Este repositorio contiene mi contribucion a a la competencia en en Kaggle [**Telstra Network Disruptions**](https://www.kaggle.com/c/telstra-recruiting-network/overview), cuyo objetivo es predecir la severidad de las interrupciones de su servicio, usando un conjunto de datos de características de sus registros de servicio, tiene la tarea de predecir si una interrupción es una falla momentánea o una interrupción total de la conectividad.

Expondre todo el proceso, desde el entendimiento de los datos, hasta mis resultados finales.

## Entendimiento del negocio
  
Telstra es una empresa de telecomunicaciones con ubicacion en Australia, es el mayor proveedor de telefonía local y larga distancia, de servicios de telefonía móvil, acceso  telefónico, conexiones inalámbricas, DSL y de acceso a Internet por cable en Australia.

Como en cualquier empresa que involucre infraestrutura computacional, mantenimiento, crecimiento en la capacidad de brindar sus servicios y tecnologia de punta, entre otras caracteristicas, suelen presentarse fallas, estas pueden ser desde ligeras hasta muy severas, al ser una empresa dedicada a las telecomunicaciones en un mundo donde la conectividad es imprescindible, tener estas fallas identificadas y sus posibles causas, es vital para repararlas en el menor tiempo posible y mantener la calidad en los servicios.

Diversas causas indiviudales o una combinacion de estas, pueden llevar a un mismo tipo de falla o serveridad en la interrupcion del servicio, teniendo las fallas clasificadas, el problema a resolver es: ¿Que factores causan cada tipo de falla? para asi poder prevenirlas.

Para esto Telstra a pone a disposicion del publico una simulacion de los datos que se generan al producirse un tipo de falla, con el objetivo de crear un modelo predictivo clasificatorio.


## Comprension de los datos

El objetivo del problema es predecir la gravedad de la falla de la red de Telstra a la vez en una ubicacion particular en funcion de los datos de registro disponibles. Los datos que se proporcionan vienen segmentados en diferentes datasets, que se unen entre si por medio de un 'id', los datasets son:

  - train.csv  Set de datos para entrenar al algortimo, contiene tres columnas: 
      -id: identificador de la falla
      -location: 
