# hash_passwords.py
from streamlit_authenticator import Hasher

plain_passwords = ["pass", "pass"]

hashed_passwords = Hasher(plain_passwords).generate()

for user, h in zip(["alice", "bob"], hashed_passwords):
    print(f"{user}: {h}")