let currentRoomId = null;
let myUsername = null;
let lastMessageId = -1;
let messageInterval = null;
let participantInterval = null;
let roomInterval = null; // For polling room list
let joinedRooms = {}; // Store joined rooms: roomId -> {name, algorithm, lastMessageId}

// UI elements
const createBtn = document.getElementById('createRoomBtn');
const createModal = document.getElementById('createModal');
const joinBtn = document.getElementById('joinRoomBtn');
const joinModal = document.getElementById('joinModal');

const messagesEl = document.getElementById('messages');
const participantsList = document.getElementById('participantsList');
const roomTitle = document.getElementById('roomTitle');
const leaveRoomBtn = document.getElementById('leaveRoomBtn');
const currentRoomDisplay = document.getElementById('currentRoomDisplay');
const searchRoomsInput = document.getElementById('searchRooms');

// small helper
function appendMessage(text, cls='') {
  const p = document.createElement('div');
  p.className = 'msg ' + cls;
  p.textContent = text;
  messagesEl.appendChild(p);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Toggle modals
createBtn.onclick = ()=> createModal.style.display = 'flex';
document.getElementById('createClose').onclick = ()=> createModal.style.display = 'none';
joinBtn.onclick = ()=> joinModal.style.display = 'flex';
document.getElementById('joinClose').onclick = ()=> joinModal.style.display = 'none';

// Create room
document.getElementById('createSubmit').onclick = async ()=>{
  const room_name = document.getElementById('roomName').value;
  const room_description = document.getElementById('roomDescription').value;
  const algorithm = document.getElementById('algorithm').value;
  const passphrase = document.getElementById('passphrase').value;
  if(!room_name || !passphrase) return alert('name+pass required');
  const res = await fetch('/api/create-room', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({room_name, room_description, algorithm, passphrase, creator:'web'})
  });
  const data = await res.json();
  if(data.room_id) {
    alert('Room created: ' + data.room_id);
    createModal.style.display = 'none';
    // Optionally auto-fill Join modal
    document.getElementById('joinRoomId').value = data.room_id;
  } else {
    alert('Error: ' + JSON.stringify(data));
  }
};

// Join room
document.getElementById('joinSubmit').onclick = async ()=>{
  const room_id = document.getElementById('joinRoomId').value.trim();
  const passphrase = document.getElementById('joinPass').value.trim();
  let username = document.getElementById('joinName').value.trim();

  if(!room_id || !passphrase) {
    alert('Please enter both Room ID and Passphrase');
    return;
  }

  // Generate username if empty
  if(!username) {
    username = 'user_' + Math.floor(Math.random() * 10000);
  }

  console.log('Attempting to join room:', {room_id, username});

  try {
    const res = await fetch('/api/join-room', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({room_id, passphrase, username})
    });
    const data = await res.json();
    console.log('Join response:', res.status, data);

    if(data.error) {
      let errorMessage = 'Join failed: ' + data.error;
      if(data.error.includes('username already taken')) {
        errorMessage += '\n\nTry a different username.';
        document.getElementById('joinName').focus();
      } else if(data.error.includes('room is full')) {
        errorMessage += '\n\nThe room has reached its maximum capacity.';
      } else if(data.error.includes('room not found')) {
        errorMessage += '\n\nPlease check the Room ID and try again.';
        document.getElementById('joinRoomId').focus();
      } else if(data.error.includes('invalid passphrase')) {
        errorMessage += '\n\nPlease check your passphrase and try again.';
        document.getElementById('joinPass').focus();
      }
      return alert(errorMessage);
    }

    myUsername = username;
    currentRoomId = room_id;
    lastMessageId = -1;
    roomTitle.textContent = `Room: ${data.name || room_id} (${data.algorithm})`;
    currentRoomDisplay.textContent = `Current Room: ${data.name || room_id}`;
    leaveRoomBtn.style.display = 'block';
    joinModal.style.display = 'none';

    const participantInfo = data.participant_count ? ` (${data.participant_count}/${data.max_participants})` : '';
    appendMessage(`You joined room "${data.name || room_id}" as ${username}${participantInfo}`, 'meta');

    // Update room info
    updateRoomInfo(room_id);
    // Start polling
    startPolling();
  } catch (error) {
    console.error('Join fetch error:', error);
    alert('Network error during join. Please check your connection and try again.');
  }
};

// Sending messages
document.getElementById('sendBtn').onclick = async ()=>{
  const input = document.getElementById('msgInput');
  const msg = input.value.trim();
  if(!msg) {
    alert('Please enter a message');
    return;
  }
  if(!currentRoomId) {
    alert('Please join a room first');
    return;
  }
  if(!myUsername) {
    alert('Please join a room first');
    return;
  }

  console.log('Sending message:', {room_id: currentRoomId, username: myUsername, message: msg, messageLength: msg.length});

  // Additional validation
  if (!currentRoomId) {
    console.error('No current room ID set');
    alert('Error: Not in a room. Please join a room first.');
    return;
  }
  if (!myUsername) {
    console.error('No username set');
    alert('Error: Username not set. Please rejoin the room.');
    return;
  }

  try {
    const requestData = {room_id: currentRoomId, username: myUsername, message: msg};
    console.log('Request data:', requestData);

    const res = await fetch('/api/send-message', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(requestData)
    });
    const data = await res.json();
    console.log('Send message response:', res.status, data);

    if (res.ok && data.success) {
      input.value = '';
      appendMessage(`${myUsername}: ${msg}`);
    } else {
      console.error('Failed to send message:', res.status, data);
      alert('Failed to send message: ' + (data.error || 'Unknown error'));
    }
  } catch (error) {
    console.error('Error sending message:', error);
    alert('Error sending message. Please check your connection.');
  }
};

function leaveRoom() {
  currentRoomId = null;
  myUsername = null;
  lastMessageId = -1;
  roomTitle.textContent = 'Select or create a room to begin';
  currentRoomDisplay.textContent = '';
  leaveRoomBtn.style.display = 'none';
  messagesEl.innerHTML = '';
  participantsList.innerHTML = '';
  appendMessage('You left the room', 'meta');
  if (messageInterval) clearInterval(messageInterval);
  if (participantInterval) clearInterval(participantInterval);
  messageInterval = null;
  participantInterval = null;
}

leaveRoomBtn.onclick = leaveRoom;

// Polling functions
function startPolling() {
  pollMessages();
  pollParticipants();
  messageInterval = setInterval(pollMessages, 2000);
  participantInterval = setInterval(pollParticipants, 5000);
}

// Initialize room polling on page load
document.addEventListener('DOMContentLoaded', function() {
  pollRooms(); // Initial load
  roomInterval = setInterval(pollRooms, 3000); // Poll every 3 seconds

  // Add search functionality
  searchRoomsInput.addEventListener('input', function() {
    const query = this.value.toLowerCase();
    const roomItems = document.querySelectorAll('#roomsList .room-card');
    roomItems.forEach(item => {
      const roomName = item.querySelector('.room-name').textContent.toLowerCase();
      if (roomName.includes(query)) {
        item.style.display = 'block';
      } else {
        item.style.display = 'none';
      }
    });
  });

  // Add Enter key support for sending messages
  document.getElementById('msgInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      document.getElementById('sendBtn').click();
    }
  });
});

async function pollMessages() {
  if (!currentRoomId) return;
  try {
    const res = await fetch(`/api/get-messages?room_id=${currentRoomId}`);
    const data = await res.json();
    if (data.messages) {
      data.messages.forEach(msg => {
        if (msg.id > lastMessageId) {
          appendMessage(`${msg.username}: ${msg.message}`);
          lastMessageId = msg.id;
        }
      });
    }
  } catch (e) {
    console.error('Poll messages error:', e);
  }
}

async function pollParticipants() {
  if (!currentRoomId) return;
  try {
    const res = await fetch(`/api/get-participants?room_id=${currentRoomId}`);
    const data = await res.json();
    if (data.participants) {
      participantsList.innerHTML = '';
      data.participants.forEach(p => {
        const li = document.createElement('li');
        li.textContent = p;
        participantsList.appendChild(li);
      });
    }
  } catch (e) {
    console.error('Poll participants error:', e);
  }
}

async function pollRooms() {
  try {
    const res = await fetch('/api/list-rooms?creator=web');
    const data = await res.json();
    if (data.rooms) {
      displayRooms(data.rooms);
    }
  } catch (e) {
    console.error('Poll rooms error:', e);
  }
}

function displayRooms(rooms) {
  const roomsListEl = document.getElementById('roomsList');
  roomsListEl.innerHTML = '';

  rooms.forEach(room => {
    const roomEl = document.createElement('div');
    roomEl.className = 'room-card';
    if (room.id === currentRoomId) {
      roomEl.classList.add('active');
    }

    roomEl.innerHTML = `
      <div class="room-name">${room.name}</div>
      <div class="room-info">
        <span class="algorithm">${room.algorithm}</span>
        <span class="participants">${room.participant_count} users</span>
      </div>
    `;

    roomEl.onclick = () => switchToRoom(room.id, room.name, room.algorithm);
    roomsListEl.appendChild(roomEl);
  });
}

// Function to update room info display
async function updateRoomInfo(roomId) {
  try {
    // Since we don't have a specific endpoint for room info, we'll use the list-rooms endpoint
    const res = await fetch('/api/list-rooms');
    const data = await res.json();
    if (data.rooms) {
      const room = data.rooms.find(r => r.id === roomId);
      if (room) {
        const roomInfoEl = document.getElementById('roomInfo');
        roomInfoEl.innerHTML = `
          <h4>Room Information</h4>
          <p><strong>Name:</strong> ${room.name}</p>
          <p><strong>Algorithm:</strong> ${room.algorithm}</p>
          <p><strong>Creator:</strong> ${room.creator}</p>
          <p><strong>Messages:</strong> ${room.message_count}</p>
          <p><strong>Participants:</strong> ${room.participant_count}</p>
        `;
      }
    }
  } catch (error) {
    console.error('Error updating room info:', error);
  }
}

function switchToRoom(roomId, roomName, algorithm) {
  if (roomId === currentRoomId) return; // Already in this room

  // If user is already in a room, leave it first
  if (currentRoomId && myUsername) {
    fetch('/api/leave-room', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({room_id: currentRoomId, username: myUsername})
    });
  }

  // Switch to new room
  currentRoomId = roomId;
  lastMessageId = -1;
  roomTitle.textContent = `Room: ${roomName} (${algorithm})`;
  currentRoomDisplay.textContent = `Current Room: ${roomName}`;
  leaveRoomBtn.style.display = 'block';

  // Clear current messages and participants
  messagesEl.innerHTML = '';
  participantsList.innerHTML = '';

  appendMessage(`Switched to room: ${roomName}`, 'meta');

  // Update room info
  updateRoomInfo(roomId);

  // Start polling for new room
  startPolling();
}
