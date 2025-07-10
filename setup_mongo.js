// MongoDB setup script
// Switch to admin database
let db = db.getSiblingDB('admin');

// Create admin user
try {
  db.createUser({
    user: 'jhowe',
    pwd: '150864',
    roles: [
      { role: 'userAdminAnyDatabase', db: 'admin' },
      { role: 'readWriteAnyDatabase', db: 'admin' },
      { role: 'dbAdminAnyDatabase', db: 'admin' }
    ]
  });
  print("✅ Admin user 'jhowe' created successfully");
} catch (e) {
  if (e.code === 51003) {
    print("ℹ️ User 'jhowe' already exists in admin database");
  } else {
    print("❌ Error creating admin user: " + e.message);
  }
}

// Switch to transcribey database
db = db.getSiblingDB('transcribey');

// Create database user
try {
  db.createUser({
    user: 'jhowe',
    pwd: '150864',
    roles: [
      { role: 'readWrite', db: 'transcribey' },
      { role: 'dbAdmin', db: 'transcribey' }
    ]
  });
  print("✅ Database user 'jhowe' created successfully for transcribey");
} catch (e) {
  if (e.code === 51003) {
    print("ℹ️ User 'jhowe' already exists in transcribey database");
  } else {
    print("❌ Error creating database user: " + e.message);
  }
}

// Create vcons collection if it doesn't exist
db.createCollection("vcons");
print("✅ Collection 'vcons' ensured to exist");

// Create performance indexes
print("🔄 Creating performance indexes...");

// Index for find_and_reserve operations (most critical)
db.vcons.createIndex(
  { "done": 1, "processed_by": 1 },
  { 
    name: "idx_reservation_status",
    background: true,
    partialFilterExpression: { "done": { $ne: true } }
  }
);
print("✅ Created reservation status index");

// Index for filename lookups (discovery/existence checks)
db.vcons.createIndex(
  { "dialog.0.filename": 1 },
  { 
    name: "idx_filename_lookup",
    background: true,
    unique: true
  }
);
print("✅ Created filename lookup index");

// Index for processed_by hostname cleanup
db.vcons.createIndex(
  { "processed_by": 1, "done": 1 },
  { 
    name: "idx_processed_by_cleanup",
    background: true,
    sparse: true
  }
);
print("✅ Created processed_by cleanup index");

// Index for general status queries
db.vcons.createIndex(
  { "done": 1, "created_at": 1 },
  { 
    name: "idx_status_created",
    background: true
  }
);
print("✅ Created status/created index");

// Test basic operations
try {
  db.vcons.insertOne({test: "connection_test", timestamp: new Date()});
  print("✅ Test document inserted successfully");
  
  var count = db.vcons.countDocuments({test: "connection_test"});
  print("✅ Found " + count + " test document(s)");
  
  db.vcons.deleteMany({test: "connection_test"});
  print("✅ Test documents cleaned up");
} catch (e) {
  print("❌ Error during test operations: " + e.message);
}

print("🎉 MongoDB setup completed!"); 