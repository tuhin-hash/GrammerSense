import axios from 'axios';

let activeBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001/api';

const client = axios.create({
  baseURL: activeBaseUrl,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  async getHealth() {
    // Probe 8001 then 8000 to dynamically detect the backend port
    const candidateUrls = [activeBaseUrl, 'http://127.0.0.1:8001/api', 'http://127.0.0.1:8000/api'];
    for (const url of candidateUrls) {
      try {
        const res = await axios.get(`${url}/health`, { timeout: 3000 });
        if (res.data?.status === 'healthy') {
          activeBaseUrl = url;
          client.defaults.baseURL = url;
          return res.data;
        }
      } catch (e) {
        // try next candidate
      }
    }
    // Final fallback attempt
    const response = await client.get('/health');
    return response.data;
  },

  async correctText(text, numCandidates = 3, numBeams = 4, temperature = 1.0) {
    const response = await client.post('/correct', {
      text,
      num_candidates: numCandidates,
      num_beams: numBeams,
      temperature,
    });
    return response.data;
  },

  async detectErrors(text) {
    const response = await client.post('/detect', { text });
    return response.data;
  },

  async getCandidates(text, numCandidates = 4) {
    const response = await client.post('/candidates', {
      text,
      num_candidates: numCandidates,
    });
    return response.data;
  },

  async evaluateDataset(datasetName = 'demo', customPairs = null, maxSamples = null) {
    const response = await client.post('/evaluate', {
      dataset_name: datasetName,
      custom_pairs: customPairs,
      max_samples: maxSamples,
    });
    return response.data;
  },
};
