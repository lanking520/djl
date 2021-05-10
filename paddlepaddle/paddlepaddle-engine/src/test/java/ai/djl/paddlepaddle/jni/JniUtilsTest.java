/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.paddlepaddle.jni;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.engine.PaddlePredictor;
import ai.djl.paddlepaddle.engine.PpNDArray;
import ai.djl.paddlepaddle.engine.PpSymbolBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class JniUtilsTest {

    @Test
    void createNDArray() {
        try (NDManager manager = NDManager.newBaseManager(null, "PaddlePaddle")) {
            NDArray array = manager.zeros(new Shape(1, 2));
            float[] expected = new float[] {0, 0};
            Assert.assertEquals(array.toFloatArray(), expected);
            array = manager.ones(new Shape(1, 2));
            expected = new float[] {1, 1};
            Assert.assertEquals(array.toFloatArray(), expected);
            expected = new float[] {1, 3, 2};
            array = manager.create(expected);
            Assert.assertEquals(array.toFloatArray(), expected);
        }
    }

    @Test
    void testMemory()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    InterruptedException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(
                                "https://resources.djl.ai/test-models/paddleOCR/mobile/det_db.zip")
                        .build();

        ZooModel<NDList, NDList> model = ModelZoo.loadModel(criteria);
        PpSymbolBlock block = ((PpSymbolBlock) model.getBlock());
        PaddlePredictor pred = block.getPredictor();
        String[] inputNames = JniUtils.getInputNames(pred);
        try (NDManager manager = NDManager.newBaseManager()) {
            for (int i = 0; i < 10000; i++) {
                PpNDArray[] inputs =
                        new PpNDArray[] {(PpNDArray) manager.ones(new Shape(1, 3, 512, 512))};
                PpNDArray[] out = JniUtils.predictorForward(pred, inputs, inputNames);
                for (PpNDArray arr : out) {
                    arr.close();
                }
                for (PpNDArray arr : inputs) {
                    arr.close();
                }
            }
        }
    }
}
