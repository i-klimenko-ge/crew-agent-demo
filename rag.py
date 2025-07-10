from typing import Dict, Any
from datetime import datetime
from rag_module import SimpleSearch

class RagSource:
    def __init__(self, description: str, source_id: str):
        self.description = description
        self.source_id = source_id

    def query(self, query_text: str) -> Dict[str, Any]:
        raise NotImplementedError("Подклассы должны реализовать метод query")

class FaissRagSource(RagSource):
    def __init__(self, source_id: str, description: str, index_path: str, texts_dict_path: str):
        self.engine = SimpleSearch(
            device="cpu",
            index_path=index_path,
            encoder_path='intfloat/multilingual-e5-large',
            texts_dict_path=texts_dict_path
        )
        self.description = description
        self.source_id = source_id
        
    def query(self, query_text: str) -> Dict[str, Any]:
        query_result = self.engine.search(query=query_text, top_n=2)

        formatted_results = "\n\n".join(
            [f"**Result {i+1}:**\n{result}" for i, result in enumerate(query_result)]
        )
        return {
            "source_id": self.source_id,
            "query": query_text,
            "results": formatted_results,
            "metadata": {"timestamp": datetime.now().isoformat()}
        }