import lancedb
import torch
import numpy as np

uri = "data/sample-lancedb"
db = lancedb.connect(uri)
tbl = db.open_table("Embeddings_db")
DB = tbl.to_pandas()
DB['id'] = DB.index + 1

embeddings = DB.Embeddings.tolist()

dataset_embeddings = torch.from_numpy(np.array(embeddings)).to(torch.float)
print(DB.columns)

