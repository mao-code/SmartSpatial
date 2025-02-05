import spacy
from collections import deque

from SmartSpatialEval.utils import (
    simple_position_mapping,
    reverse_position_mapping,
    position_phrases
)

class VLMResponseAnalyzer:
    def __init__(self):
        self.spacy = spacy.cli.download("en_core_web_sm")
        self.nlp = spacy.load("en_core_web_sm")

    # Function to find position phrases in the text
    def find_position_phrases(self, doc):
        matches = []
        for token in doc:
            found = False
            relation = ""
            for phrase in position_phrases:
                # order of position phrases matter
                if doc.text[token.idx:token.idx+len(phrase)] == phrase:
                    start = token.i
                    end = start + len(phrase.split())
                    found = True
                    relation = phrase
                    # break # Don't break cuz it may match for other phrases. e.g. in, in front of

            if found:
                matches.append((start, end, relation))

        return matches

    def find_relation(
        self,
        doc,
        position_phrases_matches,
    ):
        relations = []

        # For each matched position phrase
        for start, end, phrase in position_phrases_matches:
            # Get the simplified relation
            simplified_phrase = simple_position_mapping.get(phrase, phrase)

            # Get the first and last token of the phrase
            first_token = doc[start]
            last_token = doc[end-1] if end - 1 < len(doc) else first_token

            ##### Candidate #####
            obj_candidates = []
            first_token_head = first_token.head # Verbs like "is"
            if first_token_head.dep_ in {'conj'}:
                # When the head is conj, need to watch back again
                first_token_head = first_token_head.head

            for child in first_token_head.children:
                if child.dep_ in {'compound', 'amod', 'nsubj', "nsubjpass"}:
                    obj_candidates.append(child)

            # Search one level deeper
            for obj_cand in obj_candidates:
                for child in obj_cand.children:
                    if child.dep_ in {'compound', 'amod'}:
                        obj_candidates = [child] + obj_candidates

            # Handle for relcl
            if first_token.head.dep_ in ["relcl"]:
                obj_candidates = [first_token.head.head]

            if obj_candidates:
                # obj_text = ' '.join([o.text for o in obj_candidates] + [first_token.head.text])
                obj_text = ' '.join([o.text for o in obj_candidates])

            ##### Reference #####
            ref_candidates = [child for child in last_token.children if child.dep_ in ['pobj', 'dobj']]

            # Check modifiers for reference candidates
            extended_ref_candidates = []
            for ref in ref_candidates:
                extended_ref_candidates.append(ref)
                for child in ref.children:
                    if child.dep_ in {'compound', 'amod', 'nsubj', 'nsubjpass'}:
                        extended_ref_candidates = [child] + extended_ref_candidates

            if extended_ref_candidates:
                ref_text = ' '.join([r.text for r in extended_ref_candidates])

            if obj_candidates and ref_candidates:
                relations.append({"obj": obj_text, "pos": [simplified_phrase], "ref": ref_text})

        return relations

    def chain_relation(self, prompt_meta, relations):
        """
        Adjust relations to ensure 'center' is the central node and build shortest paths from 'obj1' and 'obj2' to 'center'.
        Shortest Path (to avoid redundent relations or paths)
        """

        center, obj_pos_pairs = prompt_meta["center"], prompt_meta["objects"]

        # Build the graph as a bidirectional (undirected) graph
        graph = {}
        for relation in relations:
            obj = relation['obj']
            ref = relation['ref']
            pos = relation['pos'][0]

            # Initialize object and ref lists if not already in graph
            if obj not in graph:
                graph[obj] = []
            if ref not in graph:
                graph[ref] = []

            # Add bidirectional (undirected) edges
            graph[obj].append((pos, ref))
            graph[ref].append((reverse_position_mapping[pos], obj))

        print("Graph: ", graph)

        # Perform BFS to find the shortest path to 'center'
        def bfs(graph, start):
            visited = {start}
            queue = deque([([start], start)])  # (path, current node) NOTE: The first is just the obj string

            while queue:
                path, current = queue.popleft()
                if current == center:
                    return path

                for pos, neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [(pos, neighbor)]
                        queue.append((new_path, neighbor))

            return None

        # Find shortest paths to 'center'
        path_objs = []
        for pair in obj_pos_pairs:
            obj, pos = pair["obj"], pair["pos"]
            path_obj = bfs(graph, obj)
            path_objs.append(path_obj)

        print("Path objs: ", path_objs)

        # Construct the results
        results = []
        for path_obj in path_objs:
            if path_obj is None or len(path_obj) <= 1: # Ensure path is not just the center itself
                continue

            results.append({
                'obj': path_obj[0],
                'pos': [p[0] for p in path_obj[1:]],
                'ref': center
            })

        print("Results: ", results)

        return results