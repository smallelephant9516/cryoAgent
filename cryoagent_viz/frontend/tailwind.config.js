/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Warm earthy palette
        cream: {
          DEFAULT: '#F7F4EF',
          light: '#FBF9F5',
        },
        terracotta: {
          DEFAULT: '#C4612F',
          hover: '#A94E22',
          tint: '#F2E3D6',
        },
        charcoal: '#1F2421',
        ink: '#1F2421',
        muted: '#5C635D',
        border: '#E7E1D7',
      },
      fontFamily: {
        serif: ['Playfair Display', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
