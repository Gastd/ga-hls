from . import treenode, defs, individual

class Diagnosis():
    """docstring for Diagnosis"""
    def __init__(self, arg = None):
        # self.arg = arg
        self.current_population = []
        self.best_diagnosis = None

    def add_new_populations(self, population: list):
        pass

    def extract_features(self):
        pass

    def diagnose(self):
        """
        Search for menor impacto na fórmula 
        """
        pass

    def remove_duplicates(self, properties: list):
        res = []
        [res.append(x) for x in properties if x not in res]
        return res

    def get_sorted_verdict(self, properties: list):
        ret = {'sats': [], 'unsats': []}

        for prop in properties:
            if not prop.madeit:
                ret['unsats'].append(chromosome)
            else:
                ret['sats'].append(chromosome)

        ret['sats'].sort(key=lambda x: x.sw_score, reverse=True)
        ret['unsats'].sort(key=lambda x: x.sw_score, reverse=True)

        return ret

    def filter(self, properties: list, per: float = .2):
        return properties[: int(len(properties) * per)]
