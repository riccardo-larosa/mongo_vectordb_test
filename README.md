# mongo_vectordb_test

* Install the required dependencies:
```bash
python -m venv myenv
source myenv/bin/activate
python -m pip install -r requirements.txt   


deactivate
```

* vector index 

 ```
 {
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "page",
      "type": "filter"
    }
  ]
}
```