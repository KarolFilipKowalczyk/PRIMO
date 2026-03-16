PAPERS = paper1-predicates paper2-bayesian paper3-hierarchy paper4-primo

.PHONY: all clean $(PAPERS)

all: $(PAPERS)

$(PAPERS):
	$(MAKE) -C $@

clean:
	@for d in $(PAPERS); do $(MAKE) -C $$d clean; done
