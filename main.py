from math import log


class ID3(object):
    def __init__(self):
        self.root = None

    def build_tree(self, dataset, root=None, target='result', exclude=[]):
        attrs = [attr for attr in dataset.header if attr not in exclude
                    and attr != target]

        if attrs:
            gains = {}
            for attr in attrs:
                gains[attr] = self.gain(dataset, attr, target=target)

            best_attr = max(gains, key=lambda x: gains[x])
            values = []
            for v in dataset[best_attr]:
                values.append(v)
            set(values)

            node = Node(best_attr, values)
            for v in node.values:
                filtered_dataset = dataset.filter(best_attr, v)
                if self.entropy(filtered_dataset, attr=best_attr, value=v,
                                 target=target) == 0:
                    node.values[v] = filtered_dataset[target][0]
                else:
                    node.values[v] = self.build_tree(filtered_dataset, node,
                                                      target, exclude+[best_attr])

            return node
        else:
            return dataset[target][0]

    def entropy(self, dataset, attr=None, value=None, target='result'):
        result = {}
        if attr:
            index_attr = dataset.header.index(attr)
        index_result = dataset.header.index(target)

        for d in dataset:
            if attr:
                if d[index_attr] == value:
                    if d[index_result] not in result:
                        result[d[index_result]] = 1
                    else:
                        result[d[index_result]] += 1
            else:
                if d[index_result] not in result:
                    result[d[index_result]] = 1
                else:
                    result[d[index_result]] += 1

        _entropy = 0.0
        total = float(sum(result.values()))

        for r in result:
            _entropy += -(result[r]/total)*log(result[r]/total, 2)

        return round(_entropy,3)

    def gain(self, dataset, attr, target='result'):
        index_attr = dataset.header.index(attr)

        values = {}
        for v in dataset[attr]:
            if v not in values:
                values[v] = 1
            else:
                values[v] += 1

        s_entropy = self.entropy(dataset, target=target)
        total = float(sum(values.values()))
        gain = s_entropy

        for v in values:
            gain += -(values[v]/total)*self.entropy(dataset, attr=attr,
                                                value=v,target=target)

        return round(gain, 3)

    def traverse(self, root=None, lvl=0):
        if not root:
            root = self.root
        if isinstance(root, Node):
            print(root.attr)
            for v in root.values:
                print(" "*lvl, end="")
                print(v, "-> ",end="")
                self.traverse(root.values[v], lvl+1)
        else:
            print(root)

    def training(self, dataset, target='result', exclude=[]):
        self.root = self.build_tree(dataset, target=target, exclude=exclude)
        self.target = target
        self.exclude = exclude

    def predict(self, data):
        node = self.root
        while True:
            value = data[node.attr]
            node = node.values[value]

            if not isinstance(node, Node):
                return node


class Node(object):
    def __init__(self, attr, values = []):
        self.attr = attr
        self.values = {}
        for v in values:
            self.values[v] = None


class Dataset(object):
    def __init__(self, data):
        self.header = data[0]
        self.data = data[1:]

    def filter(self, attr, value):
        index_attr = self.header.index(attr)
        filtered_data = [self.header]

        for d in self.data:
            if d[index_attr] == value:
                filtered_data.append(d)

        return Dataset(filtered_data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, value):
        if isinstance(value, int):
            return self.data[value]
        else:
            l = []
            index = self.header.index(value)

            for d in self.data:
                l.append(d[index])

            return l

    def __len__(self):
        return len(self.data)


def read_dataset(csvfile):
    with open(csvfile) as f:
        data = f.readlines()

    for i,d in enumerate(data):
        data[i] = d.replace("\n", "").split(";")

    dataset = Dataset(data)
    return dataset


if __name__ == '__main__':
    dataset = read_dataset('training.csv')
    model = ID3()
    model.training(dataset, target='Resultado', exclude=['Dia'])
    #model.traverse()
    data = {'Dia': 'D20',
            'Perspectiva': 'Nublado',
            'Temperatura': 'Quente',
            'Umidade': 'Alta',
            'Vento': 'Fraco'}
    print(model.predict(data))
