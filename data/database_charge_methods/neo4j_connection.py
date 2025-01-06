from neo4j import GraphDatabase
from typing import List, Dict
import os
from dotenv import load_dotenv

class Neo4jConnection:
    def __init__(self, uri=None, user=None, pwd=None):
        # Cargar variables de entorno
        load_dotenv()
        
        # Usar variables de entorno si no se proporcionan argumentos
        self._uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self._user = user or os.getenv('NEO4J_USER', 'neo4j')
        self._pwd = pwd or os.getenv('NEO4J_PASSWORD')
        
        # Inicializar driver
        self._driver = None
        self.connect()

    def connect(self):
        """Establecer conexión con la base de datos Neo4j"""
        try:
            self._driver = GraphDatabase.driver(
                self._uri, 
                auth=(self._user, self._pwd)
            )
            print("Conexión establecida exitosamente")
        except Exception as e:
            print(f"Error al conectar: {e}")
            raise

    def close(self):
        """Cerrar la conexión"""
        if self._driver:
            self._driver.close()
            print("Conexión cerrada")

    def get_session(self):
        """Obtener una sesión de la base de datos"""
        if not self._driver:
            self.connect()
        return self._driver.session()

    def execute_query(self, query, parameters=None):
        """Ejecutar una consulta genérica"""
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def create_constraints(self):
        """Crear constrains para optimizar la base de datos"""
        constraints = [
            "CREATE CONSTRAINT unique_movie IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
            "CREATE CONSTRAINT unique_user IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE"
        ]
        
        with self.get_session() as session:
            for constraint in constraints:
                session.run(constraint)