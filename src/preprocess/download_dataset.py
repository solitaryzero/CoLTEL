from datasets import load_dataset

ds = load_dataset("naist-nlp/kilt")
ds.save_to_disk('./data/kilt')

ds = load_dataset("naist-nlp/wned-wiki")
ds.save_to_disk('./data/wned-wiki')

ds = load_dataset("naist-nlp/wned-cweb")
ds.save_to_disk('./data/wned-cweb')

ds = load_dataset("naist-nlp/msnbc")
ds.save_to_disk('./data/msnbc')

ds = load_dataset("naist-nlp/aquaint")
ds.save_to_disk('./data/aquaint')

ds = load_dataset("cyanic-selkie/aida-conll-yago-wikidata")
ds.save_to_disk('./data/aida-conll-yago-wikidata')