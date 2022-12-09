import Head from 'next/head'
import { useState } from 'react';
import axios from 'axios'

export default function Home() {
  const [fileImage, setFileImage] = useState(null)
  const [responseData, setResponseData] = useState({})

  const handleSubmission = (file) => {
    const formData = new FormData();
    formData.append('file', file);
    axios.post((process.env.NEXT_PUBLIC_HOST ? `${process.env.NEXT_PUBLIC_HOST}` : '') + '/api/predict_dog/',
    formData, {
      headers: {
        'Content-Type': 'binary'
      }
  })
  .then(({ data }) => {
    setResponseData(data);
    }
  )
  .catch((error) => {
    // toast({
    //   description: t(
    //       'errors:CommandNotExec',
    //       'Command has not been executed'
    //     ),
    //   status: 'error',
    //   position: TOAST_POSITION
    // })
  });
  }

  return (
    <div>
      <Head>
        <title>Create Next App</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <input type='file' onChange={(event) => {setFileImage(event.target.files[0])}} />
        <button disabled={fileImage==null} onClick={() => {handleSubmission(fileImage)}}>Submit</button>
        {responseData && responseData?.breed?.name}
      </main>

      <footer>
      </footer>

    </div>
  )
}