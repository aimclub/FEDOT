from typing import Any, Callable, ClassVar, Iterable, List, Optional, Tuple, Type, TypeVar


Creator = TypeVar('Creator', bound=Callable[..., Any])


class Registry:
    _creators: ClassVar[List[Tuple[Callable[[Any], bool], Callable[..., Any]]]] = []
    not_found_error: ClassVar[Type[Exception]] = ValueError
    predicate_resolution_error: ClassVar[Type[Exception]] = TypeError
    not_found_message: ClassVar[str] = 'No creator registered for input: {source_type}'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '_creators' not in cls.__dict__:
            cls._creators = []

    @classmethod
    def register_creator(cls, predicate: Callable[[Any], bool]) -> Callable[[Creator], Creator]:
        def decorator(func: Creator) -> Creator:
            cls._creators.append((predicate, func))
            return func
        return decorator

    @classmethod
    def resolve_creator(cls, source_data: Any) -> Callable[..., Any]:
        return cls.resolve_registered_creator(
            creators=cls._creators,
            source_data=source_data,
            not_found_error=cls.not_found_error,
            predicate_resolution_error=cls.predicate_resolution_error,
            not_found_message=cls.not_found_message,
        )

    @classmethod
    def resolve_registered_creator(
        cls,
        creators: Iterable[Tuple[Callable[[Any], bool], Callable[..., Any]]],
        source_data: Any,
        not_found_error: Optional[Type[Exception]] = None,
        predicate_resolution_error: Optional[Type[Exception]] = None,
        not_found_message: Optional[str] = None,
    ) -> Callable[..., Any]:
        predicate_resolution_error = predicate_resolution_error or cls.predicate_resolution_error
        for predicate, creator in creators:
            result = cls.validate_creator_predicate_result(
                predicate=predicate,
                result=predicate(source_data),
                predicate_resolution_error=predicate_resolution_error,
            )
            if result:
                return creator

        raise (not_found_error or cls.not_found_error)(
            cls.format_not_found_message(
                source_data=source_data,
                not_found_message=not_found_message or cls.not_found_message,
            )
        )

    @staticmethod
    def validate_creator_predicate_result(
        predicate: Callable[[Any], bool],
        result: Any,
        predicate_resolution_error: Type[Exception] = TypeError,
    ) -> bool:
        if not isinstance(result, bool):
            raise predicate_resolution_error(
                f'Predicate {predicate.__name__} must return bool, got {type(result)}'
            )
        return result

    @staticmethod
    def format_not_found_message(source_data: Any, not_found_message: str) -> str:
        return not_found_message.format(
            source_data=source_data,
            source_type=type(source_data),
            source_type_name=type(source_data).__name__,
        )
