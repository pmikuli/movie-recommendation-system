# Uruchamianie aplikacji

Należy zacząć od instalacji zależności:
```Bash
pip install -r requirements.txt
```

Następnie należy uruchomić bazę wektorową Milvus. Baza jest uruchamiana przy pomocy Docker Compose, więc wymagane jest wcześniejsze zainstalowanie środowiska Docker. Następnie w głównym katalogu projektu należy wywołać:
```Bash
docker compose up
```
Po uruchomieniu należy otworzyć nowe okno terminala, aby nie zamykać instancji Milvus. 

Następnym krokiem jest utworzenie kolekcji w bazie wektorowej oraz wygenerowanie wektorów osadzeń filmów. Służą do tego dwa skrypty. Najpierw należy przejść do katalogu `application/two_tower_model`, a potem uruchomić odpowiednie skrypty.
```Bash
python setup_vector_database.py
python setup_item_vectors.py
```
Po wywołaniu powyższych skryptów, możliwe jest uruchomienie aplikacji. Należy przejść do katalogu `application` i w nim uruchomić komendę:
```Bash
streamlit run app.py
```
Aplikacja uruchomi się na porcie 8501 na lokalnej maszynie. 