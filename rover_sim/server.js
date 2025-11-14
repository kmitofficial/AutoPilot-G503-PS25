// =============================
// server.js
// =============================
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const path = require("path");
const cors = require("cors");
const mqtt = require("mqtt"); // ✅ NEW: MQTT bridge

// -----------------------------
// MQTT Config
// -----------------------------
const MQTT_BROKER = "mqtt://10.208.218.104";  // same IP as in dummy_r1.py
const TOPIC_VIDEO = "video/stream";
const TOPIC_STATUS = "rover/status";

// -----------------------------
// Express + Socket.IO setup
// -----------------------------
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
});

// -----------------------------
// Middleware
// -----------------------------
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "templates"))); // serve HTML

// -----------------------------
// Routes
// -----------------------------
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "templates", "index.html"));
});

app.post("/api/start_journey", (req, res) => {
  const destination = req.body.destination;
  console.log("Journey started to:", destination);
  io.emit("journey_progress", { progress: 0, destination });
  res.json({ message: "Journey started", destination });
});

// -----------------------------
// MQTT Connection
// -----------------------------
const mqttClient = mqtt.connect(MQTT_BROKER);

mqttClient.on("connect", () => {
  console.log("📡 Connected to MQTT Broker");
  mqttClient.subscribe([TOPIC_VIDEO, TOPIC_STATUS], (err) => {
    if (!err) console.log(`✅ Subscribed to: ${TOPIC_VIDEO}, ${TOPIC_STATUS}`);
  });
});

// Bridge MQTT → Socket.IO
mqttClient.on("message", (topic, message) => {
  const payload = message.toString();
  if (topic === TOPIC_VIDEO) {
    // forward frame to frontend
    io.emit("camera_frame", { frame: payload });
  } else if (topic === TOPIC_STATUS) {
    io.emit("rover_status", JSON.parse(payload));
  }
});

// -----------------------------
// Socket.IO Events
// -----------------------------
io.on("connection", (socket) => {
  console.log("🌐 Web client connected");

  socket.on("gemini_instruction", (data) => {
    console.log("Gemini instruction:", data.instruction);
    io.emit("journey_progress", data);
  });

  socket.on("disconnect", () => {
    console.log("Web client disconnected");
  });
});

// -----------------------------
// Start server
// -----------------------------
server.listen(5000, () => {
  console.log("✅ Server running at http://localhost:5000");
});
