/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct SpaceInfo {
    pub id: usize,
    pub bbox: [f64; 4],
    pub occupied: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ParkingLotSnapshot {
    pub lot: String,
    pub timestamp: String,
    pub image_b64: String,
    pub total_spaces: usize,
    pub occupied_ids: Vec<usize>,
    pub spaces: Vec<SpaceInfo>,
}
