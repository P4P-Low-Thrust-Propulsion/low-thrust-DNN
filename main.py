
def run():
    import pygmo as pg
    from Flyby import Flyby

    p = Flyby(['earth', 'mars'],
              [66281, 493],
              [0, 0], ignore_last=False, days=1, multi_revs=1)

    algo = pg.algorithm(pg.de1220(gen=10))
    archi = pg.archipelago(n=2, algo=algo, t=pg.topology(pg.ring(w=0.1)), prob=pg.problem(p), pop_size=100)
    archi.evolve(1000)

    sols = archi.get_champions_f()
    idx = sols.index(min(sols))
    p.fitness(archi.get_champions_x()[idx])
    p.print_transx()
    p.plot_trajectory()


if __name__ == "__main__":
    run()


