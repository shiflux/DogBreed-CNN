import NavBar from "../components/navbar"
import Head from "next/head"
import Script from "next/script"
import '../styles/style.css'

export default function MyApp ({ Component, pageProps }) {
    return (
        <>
            <Head>
                <link
                    href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css"
                    rel="stylesheet"
                    integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1"
                    crossOrigin="anonymous"
                />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Create Next App</title>
                <link rel="icon" href="/favicon.ico" />

                </Head>

            <Script
                src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.bundle.min.js"
                integrity="sha384-ygbV9kiqUc6oa4msXn9868pTtWMgiQaeYH7/t7LECLbyPA2x65Kgf80OJFdroafW"
                crossOrigin="anonymous"
            />

            <main className='vh-100 d-flex flex-column h-100'>
                <NavBar/>
                <Component {...pageProps} />
            </main>
        </>
    )
}