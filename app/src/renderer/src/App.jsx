import React, { useState, useEffect, useCallback } from 'react';
import { api } from './lib/api';
import Gallery from './components/Gallery.jsx';
import Chat from './components/Chat.jsx';
import CharacterEditor from './components/CharacterEditor.jsx';
import Settings from './components/Settings.jsx';
import './components/app.css';

// Views: 'gallery' | 'chat'. Editor + Settings are overlays.
export default function App() {
  const [view, setView] = useState('gallery');
  const [characters, setCharacters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);

  const [activeCharacter, setActiveCharacter] = useState(null);
  const [activeConversation, setActiveConversation] = useState(null);

  const [editorChar, setEditorChar] = useState(undefined); // undefined=closed, null=new, obj=edit
  const [showSettings, setShowSettings] = useState(false);

  const refreshCharacters = useCallback(async () => {
    try {
      const list = await api.listCharacters();
      setCharacters(list);
    } catch (e) {
      console.error('listCharacters', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshCharacters();
    api.health().then(setHealth).catch(() => setHealth({ status: 'down' }));
  }, [refreshCharacters]);

  const openChat = (character, conversationId = null) => {
    setActiveCharacter(character);
    setActiveConversation(conversationId);
    setView('chat');
  };

  const onSaveCharacter = async (data) => {
    if (editorChar && editorChar.id) {
      const updated = await api.updateCharacter(editorChar.id, data);
      // If the character being edited is the one open in chat, refresh the live
      // copy so the new persona/greeting/sampling take effect without leaving.
      if (activeCharacter?.id === updated.id) setActiveCharacter(updated);
    } else {
      await api.createCharacter(data);
    }
    setEditorChar(undefined);
    await refreshCharacters();
  };

  const onDeleteCharacter = async (id) => {
    await api.deleteCharacter(id);
    setEditorChar(undefined);
    if (activeCharacter?.id === id) setView('gallery');
    await refreshCharacters();
  };

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <div className="brand" onClick={() => setView('gallery')}>
          <span className="brand-mark" />
          <span className="brand-name">Scenario<em>Chat</em></span>
        </div>
        <div className="topbar-spacer" />
        <div className={`health-pill ${health?.status === 'ok' ? 'ok' : 'down'}`}>
          <span className="dot" />
          {health?.status === 'ok' ? health.model : 'offline'}
          {health?.sync?.enabled && <span className="sync-tag">SYNC</span>}
        </div>
        <button className="btn btn-ghost" onClick={() => setShowSettings(true)}>Settings</button>
      </header>

      <main className="app-body">
        {view === 'gallery' && (
          <Gallery
            characters={characters}
            loading={loading}
            onOpen={openChat}
            onNew={() => setEditorChar(null)}
            onEdit={(c) => setEditorChar(c)}
          />
        )}
        {view === 'chat' && activeCharacter && (
          <Chat
            character={activeCharacter}
            conversationId={activeConversation}
            onBack={() => setView('gallery')}
            onConversation={setActiveConversation}
            onEditCharacter={() => setEditorChar(activeCharacter)}
          />
        )}
      </main>

      {editorChar !== undefined && (
        <CharacterEditor
          character={editorChar}
          onClose={() => setEditorChar(undefined)}
          onSave={onSaveCharacter}
          onDelete={onDeleteCharacter}
        />
      )}

      {showSettings && <Settings health={health} onClose={() => setShowSettings(false)} />}
    </div>
  );
}
