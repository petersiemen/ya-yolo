from .context import *


def test_equals():
    clusters_1 = [
        [DofrAndPrice(dofr=1998, price=5000.0),
         DofrAndPrice(dofr=1998, price=5300.0),
         DofrAndPrice(dofr=1996, price=4000.0)],
        [DofrAndPrice(dofr=2002, price=8000.0),
         DofrAndPrice(dofr=2004, price=9000.0)]
    ]

    clusters_2 = [
        [DofrAndPrice(dofr=1998, price=5000.0),
         DofrAndPrice(dofr=1998, price=5300.0),
         DofrAndPrice(dofr=1996, price=4000.0)],
        [DofrAndPrice(dofr=2002, price=8000.0),
         DofrAndPrice(dofr=2004, price=9000.0)]
    ]

    clusters_3 = [
        [DofrAndPrice(dofr=1998, price=5000.0),
         DofrAndPrice(dofr=1998, price=5300.0)],
        [DofrAndPrice(dofr=1996, price=4000.0),
         DofrAndPrice(dofr=2002, price=8000.0),
         DofrAndPrice(dofr=2004, price=9000.0)]
    ]

    assert clusters_1 == clusters_2
    assert clusters_1 != clusters_3


def test_cluster():
    print(os.environ['HOME'])
    items = [
        DofrAndPrice(dofr=1998, price=5000.0),
        DofrAndPrice(dofr=1998, price=5300.0),
        DofrAndPrice(dofr=1996, price=4000.0),
        DofrAndPrice(dofr=2002, price=8000.0),
        DofrAndPrice(dofr=2004, price=9000.0)
    ]

    cluster(items)


def test_step():
    clusters = step([], [2, 2, 2, 5, 5, 5])
    print(clusters)
