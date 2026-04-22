## Eksloracja danych - przykładowy problem 1

### Przygotowanie środowiska

 1. Wykonaj 'fork' tego repozytorium.
 2. Sklonuj fork do lokalnego folderu
 3. Pobierz zbiór danych ze strony https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download 
 4. Utwórz w lokalnym folderze folder data/ (katalog data/ powinien być w .gitignore)
 5. Przekopiuj do folderu data/ pliki 


    ![alt text](files.png)

### Zadanie

 1. Zapoznaj się z notatnikiem judge.ipynb
 2. Utwórz roboczy branch
 3. W katalogu src/ znajdziesz plik system111333.py. Zmodyfikuj go tak, aby nazwa zawierała numer indeksu jednego z uczestników grupy projektowej. W treści tego pliku powinna znajdować się implementacja twojego systemu oceniającego.
 4. Rozwiąż zadanie
    - Przygotuj pod-zadania dotyczące tego problemu na tablicy Kanban
    - Zaplanuj daty wykonania zadań
    - Przydziel zadania członkom zespołu
    - Zaplanuj dwie iteracje
    - Termin spotkania dotyczącego mini-projektu: 17.03.2026
 5. Umieść rozwiązanie w katalogu, do którego dostęp znajdziesz na stronie kursu (tylko plik z systemem).

Termin wykonania zadania: 31.03.2026

### Rozwiązania

Aby uruchomić konkurs należy w pierwszej kolejności uzupełnić dane modeli dostarczonych przez systemy. Skopiuj do swojej lokalnej kopii repozytorium modele umieszczone w tym katalogu:
https://drive.google.com/drive/folders/1k6kbKGJ2jYi91T5Yy4MnEy7ijDDWNdx8?usp=drive_link

1. Utwórz środowisko wirtualne 
   ```python -m venv ven```
2. Zainstaluj biblioteki w odpowiednich wersjach
   ```pip install -r reqiurements.txt```
3. Uruchom konkurs
   ```python src/run.py```

Domyślnie konkurs obejmuje 10 przebiegów dla wszystkich dostarczonych systemów. Jedno uruchomienie trwa bardzo długo (kilka godzin). Możesz zmniejszyć ten czas komentując rejestrację wybranych systemów.

```
    #print('registering System156962...')
    #start = time()
    #competition.register(System156962())
    #end = time()
    #systems.append('System156962')
    #times.append(end - start)
```

Opcjonalnie możesz zmniejszyć liczbę iteracji.

```
   for i in range(10):
        start = time()
        print('--------------------------------------------------------')
        print(f'Building round robin schedule: iteration {i+1}/10')
        competition.build_round_robin()
        #run the competition - it prints out the results
        print(f'results after round robin {i+1}/10:')
        competition.compete()
        end = time()
        print(f'Round robin {i+1} completed in {end - start:.2f} seconds')
```
