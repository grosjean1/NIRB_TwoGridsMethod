# NIRB_TwoGridsMethod
# scprits Elise Grosjean 2021
Non Intrusive Reduced Basis Method

offline (Greedy algorithm) and online part
Mesh reader with Basictools
Warning: basic-tools/src must be added to the pythonpath (https://gitlab.com/drti/basic-tools/-/tree/public_master/src/BasicTools)

Two directories (One "Fine" with the snapshots (snapshot_i.vtu) i=0...ns (ns number of snapshots))
    		(One "Coarse" with the solution (snapshotH_ns.vtu))

Launch Nirb_Offline.py (with or without the numberOfmodes)
Then Nirb_Online.py
Results are the approximation in .vtu and the H1 and L2 errors