"""
Unit tests for Cyclotron Multi-Factor Ingestion, Rule Engine, and Vector Schema.
"""

import unittest
import asyncio
from watcher.schemas.vector_schema import VectorPayload, VectorSourceType, SpatialEntity
from watcher.ingestion.buffer import MultiVectorIngestionBuffer
from watcher.rules.multi_factor_engine import MultiFactorRuleEngine


class TestCyclotronMultiFactor(unittest.TestCase):
    def setUp(self):
        self.buffer = MultiVectorIngestionBuffer(max_capacity=100, max_batch_size=4)
        self.engine = MultiFactorRuleEngine("config/watcher_multi_factor_rules.yaml")

    def test_vector_schema_and_rules(self):
        payload = VectorPayload(
            vector_id="test_vec_01",
            source_type=VectorSourceType.CAMERA_STREAM,
            source_id="cam_test",
            hazard_score=0.85,
            entities=[
                SpatialEntity(label="person", confidence=0.75, bbox=[0, 0, 1, 1], class_id=0)
            ]
        )
        
        actions = self.engine.evaluate_vector(payload)
        self.assertTrue(len(actions) > 0)
        self.assertEqual(actions[0]["rule_id"], "RULE_001_ZONE_BREACH_HIGH_HAZARD")

    def test_buffer_push_and_batch(self):
        async def run_buffer_test():
            for i in range(5):
                p = VectorPayload(
                    vector_id=f"v_{i}",
                    source_type=VectorSourceType.CAMERA_STREAM,
                    source_id="cam_0",
                )
                await self.buffer.push(p)
            
            batch = await self.buffer.get_batch()
            self.assertIsNotNone(batch)
            self.assertEqual(len(batch.vectors), 4)

        asyncio.run(run_buffer_test())


if __name__ == "__main__":
    unittest.main()