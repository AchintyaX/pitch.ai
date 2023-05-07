from search_utils import PitchCollection
from pprint import pprint

test_instance = PitchCollection()
query = ["Intel unveils laser breakthrough"]
test_keywords = 'Intel, data networks'

output = test_instance(query, test_keywords)

pprint(output)