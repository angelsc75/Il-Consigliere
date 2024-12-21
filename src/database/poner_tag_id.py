import pandas as pd

# Cargar los archivos
tags_csv_path = "data/movielens/processed/tag_ok.csv"
genome_tags_csv_path = "data/movielens/genome_tags.csv"

tags_df = pd.read_csv(tags_csv_path)  # Contiene userId, movieId, tag, timestamp
genome_tags_df = pd.read_csv(genome_tags_csv_path)  # Contiene tagId, tag

# Identificar tags presentes en tag.csv pero no en genome_tags.csv
unique_tags_in_tag_csv = set(tags_df['tag'].unique())
unique_tags_in_genome_tags = set(genome_tags_df['tag'].unique())

# Tags que están en tag.csv pero no en genome_tags.csv
tags_without_id = unique_tags_in_tag_csv - unique_tags_in_genome_tags

# Tags que sí tienen id
tags_with_id = unique_tags_in_tag_csv & unique_tags_in_genome_tags

print(f"Tags sin ID en genome_tags.csv: {len(tags_without_id)}")
print(f"Tags con ID en genome_tags.csv: {len(tags_with_id)}")

# Guardar los tags sin ID para inspección
pd.DataFrame(list(tags_without_id), columns=['tag']).to_csv("tags_without_id.csv", index=False)

# Crear un DataFrame para tags sin ID
tags_without_id_df = pd.DataFrame(list(tags_without_id), columns=['tag'])

# Generar un nuevo rango de tagId para estos tags
start_id = genome_tags_df['tagId'].max() + 1  # Empezar donde termina genome_tags.csv
tags_without_id_df['tagId'] = range(start_id, start_id + len(tags_without_id_df))

# Guardar los nuevos tags con ID
tags_without_id_df.to_csv("new_tags_with_id.csv", index=False)

print("Nuevos tags con ID generados y guardados en 'new_tags_with_id.csv'")
# Combinar los DataFrames
combined_tags_df = pd.concat([genome_tags_df, tags_without_id_df], ignore_index=True)

# Guardar el archivo combinado
combined_tags_df.to_csv("combined_tags.csv", index=False)

print("Archivo combinado de tags guardado como 'combined_tags.csv'")