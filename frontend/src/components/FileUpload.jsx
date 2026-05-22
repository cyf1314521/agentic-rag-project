/** PDF 拖拽/选择上传，调用 /api/files/upload */
import { useState, useRef } from 'react';
import { Upload, X, FileText, Loader2 } from 'lucide-react';
import { uploadFiles } from '../api';

export default function FileUpload({ sessionId, onUploaded, onSessionCreated }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState([]);
  const inputRef = useRef(null);

  const handleFiles = async (fileList) => {
    const pdfs = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length === 0) return;

    setUploading(true);
    setResults([]);
    try {
      const res = await uploadFiles(pdfs, sessionId || null);
      setResults(res.files || []);
      if (res.session_id && onSessionCreated) onSessionCreated(res.session_id);
      if (onUploaded) onUploaded();
    } catch (err) {
      setResults([{ filename: 'Upload failed', status: 'error', detail: err.message }]);
    }
    setUploading(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="p-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
          dragging ? 'border-black bg-gray-50' : 'border-gray-200 hover:border-gray-300'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        {uploading ? (
          <Loader2 size={24} className="mx-auto text-gray-400 animate-spin" />
        ) : (
          <Upload size={24} className="mx-auto text-gray-400" />
        )}
        <p className="text-sm text-gray-500 mt-2">
          {uploading ? 'Processing...' : 'Drop PDF files here or click to browse'}
        </p>
        {sessionId ? (
          <p className="text-xs text-gray-400 mt-1">Files are linked to this chat only.</p>
        ) : (
          <p className="text-xs text-amber-600 mt-1">Start or select a chat first to scope uploads.</p>
        )}
      </div>

      {results.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {results.map((r, i) => {
            const isOk = r.status === 'ok';
            const isDup = r.status === 'duplicate';
            const boxClass = isOk
              ? 'bg-green-50 text-green-700'
              : isDup
                ? 'bg-blue-50 text-blue-800'
                : 'bg-red-50 text-red-700';
            return (
            <div key={i} className={`flex items-start gap-2 text-xs px-3 py-2 rounded-lg ${boxClass}`}>
              <FileText size={13} className="mt-0.5 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium">{r.filename}</div>
                {isOk && <div>{r.chunk_count} chunks</div>}
                {isDup && (
                  <div>
                    已在库中（{r.paper_id}），已绑定本会话，可直接提问。
                  </div>
                )}
                {!isOk && !isDup && <div>{r.detail}</div>}
              </div>
            </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
