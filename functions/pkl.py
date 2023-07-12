import pickle

def store_data(data, file_name):
    dbfile = open(file_name, 'ab')
    pickle.dump(data, dbfile)
    dbfile.close()
  
def load_data(file_name):
    dbfile = open(file_name, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db

def hello():
  print("hello")