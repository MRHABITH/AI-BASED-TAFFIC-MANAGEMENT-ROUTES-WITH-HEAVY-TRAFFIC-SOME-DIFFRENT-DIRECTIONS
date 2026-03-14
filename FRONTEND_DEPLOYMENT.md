# Traffic Management System - Vercel Frontend Deployment

## 🚀 Deployment Steps

### 1. Prerequisites

- Vercel account: https://vercel.com/signup
- Backend API URL (from HF Spaces)

### 2. Connect to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel
```

### 3. Environment Variables

In Vercel Dashboard → Project Settings → Environment Variables:

**Development:**
- `VITE_API_BASE_URL`: `http://localhost:8000/api`
- `VITE_API_TIMEOUT`: `60000`

**Production:**
- `VITE_API_BASE_URL`: `https://[username]-[spacename].hf.space/api`
- `VITE_API_TIMEOUT`: `60000`

### 4. Automatic Deployment

Enable "Automatic Deployments" in Vercel:
- Connect your GitHub repository
- Set production branch to `main`
- Vercel will auto-deploy on push

### 5. Custom Domain

Add your custom domain in Vercel → Project Settings → Domains

## 📊 Build Optimization

The project is pre-configured with:
- Code splitting for faster loading
- Minification and tree-shaking
- Bundle analysis support
- Optimized chunk loading

Run build analysis:
```bash
npm run analyze
```

## 🔒 Performance Tips

1. **Enable caching**: Vercel automatically caches static assets
2. **Monitor bundle size**: Check analytics in Vercel dashboard
3. **API rate limiting**: Implement on backend if needed
4. **Image optimization**: Consider CDN for large images

## 🐛 Troubleshooting

**Issue**: API connection fails
- Check `VITE_API_BASE_URL` environment variable
- Verify backend is running and accessible
- Check CORS settings on backend

**Issue**: Slow build
- Vercel caches dependencies automatically
- Check for unnecessary imports
- Run `npm run analyze` to find large dependencies

**Issue**: Blank page on production
- Check browser console for errors
- Verify environment variables are set
- Check that API endpoints are correct

## 📝 Monitoring

Monitor your Vercel deployment:
- **Vercel Analytics**: https://vercel.com/docs/analytics
- **Speed Insights**: Check CLS, LCP, FID
- **Error tracking**: Enable error tracking in settings

## 🔄 Continuous Integration

Vercel includes automatic:
- Preview deployments on PRs
- Rollback to previous versions
- GitOps integration
- Zero-downtime deployments
