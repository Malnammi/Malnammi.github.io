(define (domain gridworld-nav)
  (:requirements :strips :typing)
  (:types cell)
  (:predicates
    (at ?c - cell)
    (adj ?from - cell ?to - cell)
    (free ?c - cell))
  (:action move
    :parameters (?from - cell ?to - cell)
    :precondition (and (at ?from) (adj ?from ?to) (free ?to))
    :effect (and (not (at ?from)) (at ?to)))
)
