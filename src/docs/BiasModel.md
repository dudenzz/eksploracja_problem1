# Metoda nr 1 - BIAS MODEL
## Opis podejścia - Bias-Based Collaborative Filtering
Głównym założeniem tej metody jest system rekomendacji opierający się na odchyleniach. Zamiast zwykłego wyciągania średniej, próbujemy przewidzieć ocene z uwzględnieniem:
- ogólnej średnii,
- tak zwany charaktere użytkownika,
- jakość filmu

## Funkcje z pliku system155198.py
`calculate_global_mean` - odniesienie do całego systemu, czyli wyliczenie całości średniej globalnej.
`calculate_user_bias` - sprawdzenie, o ile średnia ocen konretnego usera różni się od średniej globalnej.
`calculate_movie_bias` - to samo, tylko dotyczy filmów
`clamp_raiting` - tutaj znajduje sie zabezpieczenie, tzn. suma biasów nie może wyskoczyć poza naszą skalę.
`rate` - zlicza względem naszej funkcji: średnia + bias usera + bias użytkownika. Główny punkt algorytmu.

## Znalezione źródła
[Artykuł PMC - 10 mar 2026]
https://pmc.ncbi.nlm.nih.gov/articles/PMC12208497/
- Zbliżona metoda do naszego podejścia, pokazuje że metody korygujące błąd (u nas lambda), dają relanie lepsze rekomendacje niż czyste systemy bez korekty

[Artykuł ARXIV - 11 mar 2026]
https://arxiv.org/pdf/2203.00376
- Podobny realizowany problem implementacji rekomendacji oraz wybrane sposoby radzenia sobie z nimi - skłonności użytkownika do popularności/jego preferencji

[Artykuł CEUR-WS - 11 mar 2026]
https://ceur-ws.org/Vol-2440/paper6.pdf
- Analiza problemu zamiast wzorów, podejście bardziej pod kątem koncepcyjnym

[Artykuł ACM - 12 mar 2026]
https://dl.acm.org/doi/abs/10.1145/3437963.3441820
- Artykuł ten stanowi bardziej rozwinięta wersje naszego podejścia