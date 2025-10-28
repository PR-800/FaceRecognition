from cryptography.fernet import Fernet
import json
import numpy as np
import base64

class EmbeddingEncryption:
    def __init__(self, key=None):
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key
        self.cipher = Fernet(self.key)
        
    def get_key(self):
        return self.key.decode('utf-8')
    
    # vector -> encrypted string
    def encrypt_embedding(self, embedding):
        embedding_list = embedding.tolist()
        json_str = json.dumps(embedding_list)
        
        encrypted = self.cipher.encrypt(json_str.encode('utf-8'))
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    # decrypted string -> vector
    def decrypt_embedding(self, encrypted_str):
        encrypted = base64.b64decode(encrypted_str.encode('utf-8'))
        
        decrypted = self.cipher.decrypt(encrypted)
        
        embedding_list = json.loads(decrypted.decode('utf-8'))
        return np.array(embedding_list)
    
    # encrypted embedding dict
    def encrypt_database(self, db_embeddings):
        encrypted_db = {}
        for person, embedding in db_embeddings.items():
            encrypted_db[person] = self.encrypt_embedding(embedding)
        return encrypted_db
    
    # decrypted embedding dict
    def decrypt_database(self, encrypted_db):
        decrypted_db = {}
        for person, encrypted in encrypted_db.items():
            decrypted_db[person] = self.decrypt_embedding(encrypted)
        return decrypted_db

# Encrpytion demo
if __name__ == "__main__":
    encryptor = EmbeddingEncryption()
    print(f"Encryption Key: {encryptor.get_key()}")
    
    example_embedding = np.random.rand(128)  # 128-dim vector like FaceNet
    print(f"Original embedding (first 5 values): {example_embedding[:5]}")
    
    encrypted = encryptor.encrypt_embedding(example_embedding)
    print(f"\nEncrypted: {encrypted[:50]}...")
    
    decrypted = encryptor.decrypt_embedding(encrypted)
    print(f"\nDecrypted (first 5 values): {decrypted[:5]}")
    
    print(f"\nMatch: {np.allclose(example_embedding, decrypted)}")
