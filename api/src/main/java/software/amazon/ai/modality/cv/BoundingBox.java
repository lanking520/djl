/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package software.amazon.ai.modality.cv;

/** An interface representing a bounding box for the detected object. */
public interface BoundingBox {

    /**
     * Gets the bounding <code>Rectangle</code> of this <code>BoundingBox</code>.
     *
     * @return a new <code>Rectangle</code> for this <code>BoundingBox</code>.
     */
    Rectangle getBounds();

    /**
     * Returns an iterator object that iterates along the <code>BoundingBox</code> boundary and
     * provides access to the geometry of the <code>BoundingBox</code> outline.
     *
     * @return a <code>PathIterator</code> object, which independently traverses the geometry of the
     *     <code>BoundingBox</code>.
     */
    PathIterator getPath();
}
