import itertools

ks = [2,4,8,16,32,64,128]
epss = [0.03]
hgs = [hg.strip() for hg in open('allhgs.txt', 'r')]
instances = list(itertools.product(hgs, ks, epss))

category_map = {
		"mtx.hgr" : "SPM",
		"cnf.dual.hgr" : "Dual",
		"cnf.primal.hgr" : "Primal",
		"cnf.hgr" : "Literal",
		"dac2012" : "DAC",
		"ISPD98_ibm" : "ISPD98"
}

def get_category(hg):
	for k,v in category_map.items():
		if k in hg:
			return k,v
	print("Error.", hg, "could not be mapped to category")
	return None

instance_grouper = ["graph", "k", "epsilon"]

ps_hgs = [hg.strip() for hg in open('parameter_tuning_set.txt', 'r')]
ps_instances = list(itertools.product(ps_hgs, ks, epss))

setB = [hg.strip() for hg in open('setB.txt', 'r')]
setB_instances = list(itertools.product(setB, ks, epss))
