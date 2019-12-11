from .context import *


def test_cluster():
    items = [
        DofrAndPrice(dofr=1998, price=5000.0),
        DofrAndPrice(dofr=1998, price=5300.0),
        DofrAndPrice(dofr=1996, price=4000.0),
        DofrAndPrice(dofr=2002, price=8000.0),
        DofrAndPrice(dofr=2004, price=9000.0)
    ]


    cluster(items)