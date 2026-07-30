"""Deterministic generation of the synthetic micro-world."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence, TypeVar


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Seed:
    """A reproducible source of keyed pseudo-random decisions.

    Decisions are keyed instead of being read from one mutable random stream.
    Adding a new generated field therefore does not change existing fields.
    """

    value: str

    def __init__(self, value: str | int) -> None:
        object.__setattr__(self, "value", str(value))

    def _number(self, key: str) -> int:
        payload = f"micro-world-v1\0{self.value}\0{key}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest(), "big")

    def choice(self, key: str, values: Sequence[T]) -> T:
        if not values:
            raise ValueError(f"cannot choose {key!r} from an empty collection")
        return values[self._number(key) % len(values)]

    def happens(self, key: str, *, numerator: int, denominator: int) -> bool:
        if denominator <= 0 or not 0 <= numerator <= denominator:
            raise ValueError("probability must satisfy 0 <= numerator <= denominator")
        return self._number(key) % denominator < numerator

    def ordered(self, key: str, values: Sequence[T]) -> list[T]:
        return sorted(values, key=lambda value: self._number(f"{key}\0{value}"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.value.encode()).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class Person:
    id: str
    name: str
    birth_date: date
    occupation: str
    birth_city: str
    residence_city: str

    @property
    def moved(self) -> bool:
        return self.birth_city != self.residence_city

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "birth_date": {
                "day": self.birth_date.day,
                "month": self.birth_date.month,
                "year": self.birth_date.year,
            },
            "occupation": self.occupation,
            "birth_city": self.birth_city,
            "residence_city": self.residence_city,
        }


@dataclass(frozen=True, slots=True)
class Fact:
    id: str
    subject_id: str
    relation: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "relation": self.relation,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class World:
    id: str
    seed: Seed
    people: tuple[Person, ...]

    @property
    def facts(self) -> tuple[Fact, ...]:
        facts: list[Fact] = []
        for person in self.people:
            values = {
                "birth_date": person.birth_date.isoformat(),
                "occupation": person.occupation,
                "birth_city": person.birth_city,
                "residence_city": person.residence_city,
            }
            facts.extend(
                Fact(
                    id=f"{person.id}.{relation}",
                    subject_id=person.id,
                    relation=relation,
                    value=value,
                )
                for relation, value in values.items()
            )
        return tuple(facts)

    def person(self, person_id: str) -> Person:
        for person in self.people:
            if person.id == person_id:
                return person
        raise KeyError(person_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "world_id": self.id,
            "seed": self.seed.value,
            "people": [person.to_dict() for person in self.people],
            "facts": [fact.to_dict() for fact in self.facts],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(frozen=True, slots=True)
class WorldSourceData:
    names: tuple[str, ...]
    dates: tuple[date, ...]
    professions: tuple[str, ...]
    towns: tuple[str, ...]

    @classmethod
    def load(cls, data_dir: Path = DEFAULT_DATA_DIR) -> "WorldSourceData":
        return cls(
            names=tuple(_read_unique_lines(data_dir / "strange_names.txt")),
            dates=tuple(
                date.fromisoformat(value)
                for value in _read_unique_lines(data_dir / "dates.txt")
            ),
            professions=tuple(_read_unique_lines(data_dir / "professions.txt")),
            towns=tuple(_read_unique_lines(data_dir / "towns.txt")),
        )


class WorldGenerator:
    def __init__(self, source_data: WorldSourceData | None = None) -> None:
        self.source_data = source_data or WorldSourceData.load()
        if len(self.source_data.towns) < 2:
            raise ValueError("at least two towns are required to model relocation")

    def generate(self, *, seed: Seed | str | int, people_count: int) -> World:
        if people_count < 0:
            raise ValueError("people_count must be non-negative")
        seed = seed if isinstance(seed, Seed) else Seed(seed)
        names = seed.ordered("person_names", self.source_data.names)
        if people_count > len(names):
            raise ValueError(
                f"requested {people_count} people, but only "
                f"{len(names)} unique names are available"
            )

        people = tuple(
            self._generate_person(seed, index, name)
            for index, name in enumerate(names[:people_count], start=1)
        )
        return World(id=f"world_{seed.fingerprint}", seed=seed, people=people)

    def _generate_person(self, seed: Seed, index: int, name: str) -> Person:
        person_id = f"person_{index:04d}"
        key = person_id
        birth_city = seed.choice(f"{key}.birth_city", self.source_data.towns)
        residence_city = birth_city

        if seed.happens(f"{key}.moved", numerator=1, denominator=3):
            other_towns = tuple(
                town for town in self.source_data.towns if town != birth_city
            )
            residence_city = seed.choice(f"{key}.residence_city", other_towns)

        return Person(
            id=person_id,
            name=name,
            birth_date=seed.choice(f"{key}.birth_date", self.source_data.dates),
            occupation=seed.choice(
                f"{key}.occupation", self.source_data.professions
            ),
            birth_city=birth_city,
            residence_city=residence_city,
        )


def _read_unique_lines(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"{path} contains no values")
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate values")
    return values
