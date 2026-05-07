use bytes::Bytes;
use serde::{Deserialize, Serialize};

pub const DEFAULT_SOCKET_PATH: &str = "/tmp/vision-rs-parking-garage.sock";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ParkingEvent {
    Heartbeat {
        sequence: u64,
    },
    Occupancy {
        occupied: u32,
        capacity: u32,
    },
    VehicleSeen {
        id: u64,
        bay: String,
    },
}

pub fn encode_event(event: &ParkingEvent) -> Result<Bytes, postcard::Error> {
    postcard::to_allocvec(event).map(Bytes::from)
}

pub fn decode_event(bytes: &[u8]) -> Result<ParkingEvent, postcard::Error> {
    postcard::from_bytes(bytes)
}
