# CS2 AI Autoshot
# Искусственный интеллект для автоматической стрельбы в Counter-Strike 2

## О программе
Сверточная нейросеть, обученная для сегментации изображениях в CS2. Предусмотрено 3 класса объектов: 0 - Ничего, 1 - Спецназ, 2 - Террорист. Выборка состоит из 4 изображений спецназа и такого же количества террористов (карта Mirage), для увеличения объема выборки применялась аугментация (поворот изображений + отражение). В финальная выборка состояла из 96 изображений разрешения 512x512.

## Как работает программа
1. Каждые 100мс программа делает снимок экрана,
2. из снимка выбирается квадрат 512x512 пикселей в центре экрана,
3. нейросеть строит маску для данного квадрата,
4. из маски выбирается малый сегмент в центре размером 10x10 пикселей,
5. если большинство пикселей сегмента принадлежат заданному классу и уверенность нейросети больше 50%, то принимается решение о выстреле.

## Заметки
1. Исследования показали, что разрешение 128x128 совершенно недостаточно для поиска объектов в CS2, 
2. разрешение 256x256 приемлемо, но в изображение попадает мало объектов,
3. разрешение 256x256, полученное в результате сжатия изображения 512x512, показало слабые результаты - из-за размыливания объекты часто путаются,
4. разрешение 512x512 показывает хорошие результаты распознавания объектов даже на малой выборке, но требует 90-100 мс на 1 кадр.
5. 
![1](https://github.com/DaniilKlyukin/CS2_AI_Autoshot/assets/32903150/4dcc63d9-dddf-422e-9fb6-d28a6e65fcf8) </br>
![2](https://github.com/DaniilKlyukin/CS2_AI_Autoshot/assets/32903150/26d9b548-fbd9-4af7-82e2-bd14bbf35be0) </br>
![3](https://github.com/DaniilKlyukin/CS2_AI_Autoshot/assets/32903150/52c593bb-baef-45bd-a628-5ece87afde41) </br>
![4](https://github.com/DaniilKlyukin/CS2_AI_Autoshot/assets/32903150/21019803-ab27-44b5-b0a4-672b3fc03aac) </br>
![5](https://github.com/DaniilKlyukin/CS2_AI_Autoshot/assets/32903150/5310228e-35cf-479d-a078-da1129351d4b) </br>
![6](https://github.com/DaniilKlyukin/CS2_AI_Autoshot/assets/32903150/fc3ff9ec-ede2-444e-b688-dbe41a8e6782) </br>
![7](https://github.com/DaniilKlyukin/CS2_AI_Autoshot/assets/32903150/865aede4-4677-4542-bad1-2944e17a858f) </br>
![8](https://github.com/DaniilKlyukin/CS2_AI_Autoshot/assets/32903150/9cec139b-438f-4b74-80ca-1766be3c1f75) </br>
