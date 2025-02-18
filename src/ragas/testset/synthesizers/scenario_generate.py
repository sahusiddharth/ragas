from typing import List, Optional
from langchain_core.callbacks import Callbacks

# Import the base scenario types and synthesizers:
from ragas.testset.synthesizers.base import BaseScenario
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
    SingleHopScenario,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample


class UserScenarioTestsetGenerator:
    """
    A helper class that creates a Testset from a list of user-provided scenarios.
    It supports both single-hop and multi-hop scenarios.
    """

    def __init__(self, llm):
        # Use the same LLM instance as used elsewhere in your pipeline.
        self.llm = llm

    async def generate_from_user_scenarios(
        self,
        scenarios: List[BaseScenario],
        callbacks: Optional[Callbacks] = None,
    ) -> Testset:
        """
        Generate a Testset from a list of user-specified scenarios.

        Parameters
        ----------
        scenarios : List[BaseScenario]
            A list of scenarios (which may be instances of SingleHopScenario or MultiHopScenario)
            provided by the user.
        callbacks : Optional[Callbacks]
            Optional callbacks for logging/tracking the generation process.

        Returns
        -------
        Testset
            A Testset containing the generated samples.
        """
        samples = []
        additional_info = []

        # Process each scenario with the appropriate synthesizer
        for scenario in scenarios:
            if isinstance(scenario, SingleHopScenario):
                synthesizer = SingleHopSpecificQuerySynthesizer(llm=self.llm)
            elif isinstance(scenario, MultiHopScenario):
                synthesizer = MultiHopSpecificQuerySynthesizer(llm=self.llm)
            else:
                raise ValueError(f"Unsupported scenario type: {type(scenario)}")

            sample = await synthesizer.generate_sample(scenario, callbacks=callbacks)
            samples.append(sample)
            additional_info.append({"synthesizer_name": synthesizer.name})

        # Wrap the samples into TestsetSample objects
        testset_samples = [
            TestsetSample(eval_sample=s, synthesizer_name=info["synthesizer_name"])
            for s, info in zip(samples, additional_info)
        ]
        return Testset(samples=testset_samples)
